import setGPU
import os
import glob
import sys
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score
import resnet_v1_eembc
import yaml
import csv
import setGPU
#from keras_flops import get_flops #(different flop calculation)
import kerop
from tensorflow.keras.datasets import cifar10
from train import get_lr_schedule_func, yaml_load
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import KLDivergence, CategoricalCrossentropy

class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        teacher_loss_fn,
        distillation_loss_fn,
        alpha=0,
        beta=0.1,
        gamma=0.9,
        temperature=2,
    ):
        """ Configure the distiller.
        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to teacher_loss_fn 
            beta: weight to student_loss_fn
            gamma: weight to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.teacher_loss_fn = teacher_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            teacher_loss = self.teacher_loss_fn(y, teacher_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(tf.math.log(teacher_predictions) / self.temperature, axis=1),
                tf.nn.softmax(tf.math.log(student_predictions) / self.temperature, axis=1),
            )
            loss = self.alpha * teacher_loss + self.beta * student_loss + self.gamma * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, 
             "distillation_loss": distillation_loss,
             "teacher_loss": teacher_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_student = self.student(x, training=False)
        y_teacher = self.teacher(x, training=False)

        # Calculate the losses
        student_loss = self.student_loss_fn(y, y_student)
        teacher_loss = self.teacher_loss_fn(y, y_teacher)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_student)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss,
                        "teacher_loss": teacher_loss})
        return results

    def call(self, inputs):
        return self.student(inputs)


def main(args):

    # parameters
    input_shape = [32,32,3]
    num_classes = 10
    config = yaml_load(args.config)
    num_filters = config['model']['filters']
    kernel_sizes = config['model']['kernels']
    strides = config['model']['strides']
    l1p = float(config['model']['l1'])
    l2p = float(config['model']['l2'])
    skip = bool(config['model']['skip'])
    avg_pooling = bool(config['model']['avg_pooling'])
    batch_size = config['fit']['batch_size']
    num_epochs = config['fit']['epochs']
    verbose = config['fit']['verbose']
    patience = config['fit']['patience']
    save_dir = config['save_dir']
    model_name = config['model']['name']
    loss = config['fit']['compile']['loss']
    model_file_path = os.path.join(save_dir, 'model_distill_best_weights.h5')
    model_load_path = os.path.join(save_dir, 'model_best.h5')

    # quantization parameters
    if 'quantized' in model_name:
        logit_total_bits = config["quantization"]["logit_total_bits"]
        logit_int_bits = config["quantization"]["logit_int_bits"]
        activation_total_bits = config["quantization"]["activation_total_bits"]
        activation_int_bits = config["quantization"]["activation_int_bits"]
        alpha = config["quantization"]["alpha"]
        use_stochastic_rounding = config["quantization"]["use_stochastic_rounding"]
        logit_quantizer = config["quantization"]["logit_quantizer"]
        activation_quantizer = config["quantization"]["activation_quantizer"]

    # optimizer
    optimizer = getattr(tf.keras.optimizers,config['fit']['compile']['optimizer'])
    initial_lr = config['fit']['compile']['initial_lr']
    lr_decay = config['fit']['compile']['lr_decay']

    # load dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train, X_test = X_train/256., X_test/256.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # define data generator
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        #brightness_range=(0.9, 1.2),
        #contrast_range=(0.9, 1.2)
    )

    # run preprocessing on training dataset
    datagen.fit(X_train)

    kwargs = {'input_shape': input_shape,
              'num_classes': num_classes,
              'num_filters': num_filters,
              'kernel_sizes': kernel_sizes,
              'strides': strides,
              'l1p': l1p,
              'l2p': l2p,
              'skip': skip,
              'avg_pooling': avg_pooling}

    # pass quantization params
    if 'quantized' in model_name:
        kwargs["logit_total_bits"] = logit_total_bits
        kwargs["logit_int_bits"] = logit_int_bits
        kwargs["activation_total_bits"] = activation_total_bits
        kwargs["activation_int_bits"] = activation_int_bits
        kwargs["alpha"] = None if alpha == 'None' else alpha
        kwargs["use_stochastic_rounding"] = use_stochastic_rounding
        kwargs["logit_quantizer"] = logit_quantizer
        kwargs["activation_quantizer"] = activation_quantizer

    # define model
    student = getattr(resnet_v1_eembc,model_name)(**kwargs)
    #student.load_weights(model_load_path)

    # print model summary
    print('#################')
    print('# MODEL SUMMARY #')
    print('#################')
    print(student.summary())
    print('#################') 

    # load baseline
    teacher = load_model('resnet_v1_eembc/model_best.h5')
    teacher._name = 'teacher'
    for layer in teacher.layers:
        layer._name = 'teacher_' + layer.name

    # initialize and compile distiller
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=optimizer(learning_rate=initial_lr),
        metrics=['accuracy'],
        student_loss_fn=CategoricalCrossentropy(),
        teacher_loss_fn=CategoricalCrossentropy(),
        distillation_loss_fn=CategoricalCrossentropy(),#KLDivergence(),
        alpha=0,
        beta=0.1,
        gamma=0.9,
        temperature=2,
    )

    # callbacks
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

    lr_schedule_func = get_lr_schedule_func(initial_lr, lr_decay)

    callbacks = [ModelCheckpoint(model_file_path, monitor='val_accuracy', verbose=verbose, save_best_only=True, save_weights_only=True),
                 EarlyStopping(monitor='val_accuracy', patience=patience, verbose=verbose, restore_best_weights=True),
                 LearningRateScheduler(lr_schedule_func, verbose=verbose),
    ]

    # train
    history = distiller.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        epochs=num_epochs,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks,
                        verbose=verbose)


    # restore "best" model
    distiller.load_weights(model_file_path)

    # get predictions
    y_pred = distiller.predict(X_test)

    # evaluate with test dataset and share same prediction result
    evaluation = distiller.evaluate(X_test, y_test)
    
    auc = roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr')

    print('Model test accuracy = %.4f' % evaluation[0])
    print('Model test weighted average AUC = %.4f' % auc)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default = "baseline.yml", help="specify yaml config")

    args = parser.parse_args()

    main(args)

