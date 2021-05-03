import os
import glob
import sys
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score
from resnet_v1_eembc import resnet_v1_eembc, resnet_v1_eembc_quantized
import yaml
import csv
import setGPU
import kerop
from train import get_lr_schedule_func
import kerastuner
from tensorflow.keras.datasets import cifar10

filter_space = [2, 4, 8, 16, 32]
kernelsize_space = [1, 2, 3, 4]

# define cnn model
def build_model(hp):
    # default 3 stacks
    hp_filters0_0 = hp.Choice('filters0_0', filter_space)
    hp_filters0_1 = hp.Choice('filters0_1', filter_space)
    hp_filters1_0 = hp.Choice('filters1_0', filter_space)
    hp_filters1_1 = hp.Choice('filters1_1', filter_space)
    hp_filters2_0 = hp.Choice('filters2_0', filter_space)
    hp_filters2_1 = hp.Choice('filters2_1', filter_space)
    hp_kernelsize0_0 = hp.Choice('kernelsize0_0', kernelsize_space)
    hp_kernelsize0_1 = hp.Choice('kernelsize0_1', kernelsize_space)
    hp_kernelsize0_2 = hp.Choice('kernelsize0_2', kernelsize_space)
    hp_kernelsize1_0 = hp.Choice('kernelsize1_0', kernelsize_space)
    hp_kernelsize1_1 = hp.Choice('kernelsize1_1', kernelsize_space)
    hp_kernelsize1_2 = hp.Choice('kernelsize1_2', kernelsize_space)
    hp_kernelsize2_0 = hp.Choice('kernelsize2_0', kernelsize_space)
    hp_kernelsize2_1 = hp.Choice('kernelsize2_1', kernelsize_space)
    hp_kernelsize2_2 = hp.Choice('kernelsize2_2', kernelsize_space)

    hp_strides0 = hp.Choice('strides0', ['111', '211', '244', '311', '334', '343', '344', '411', '424', '433', '434', '442', '443', '444'])
    hp_strides1 = hp.Choice('strides1', ['111', '122', '133', '144', '212', '224', '313', '414'])
    hp_strides2 = hp.Choice('strides2', ['111', '122', '133', '144', '212', '224', '313', '414'])
    
    model = resnet_v1_eembc_quantized(input_shape=[32, 32, 3], num_classes=10, l1p=0, l2p=1e-4,
                            num_filters=[hp_filters0_0, hp_filters0_1, 
                                         hp_filters1_0, hp_filters1_1,
                                         hp_filters2_0, hp_filters2_1],
                            kernel_sizes=[hp_kernelsize0_0, hp_kernelsize0_1, hp_kernelsize0_2,
                                          hp_kernelsize1_0, hp_kernelsize1_1, hp_kernelsize1_2,
                                          hp_kernelsize2_0, hp_kernelsize2_1, hp_kernelsize2_2],
                            strides=[hp_strides0, 
                                     hp_strides1, 
                                     hp_strides2], 
                            avg_pooling=False,
                            skip=False,
                            logit_total_bits=8, logit_int_bits=2, activation_total_bits=8, activation_int_bits=2,
                            alpha=1, use_stochastic_rounding=False,
                            logit_quantizer = 'quantized_bits', activation_quantizer = 'quantized_relu',
                        )
    # compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main(args):

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    num_classes = 10

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    
    # define data generator                                                                                                     
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )

    tunerClass = getattr(kerastuner.tuners,args.tuner)
    hp = kerastuner.HyperParameters()
    if args.stacks==2:
        hp.Fixed('filters2_0', 0)
        hp.Fixed('filters2_1', 0)
        hp.Fixed('kernelsize2_0', 0)
        hp.Fixed('kernelsize2_1', 0)
        hp.Fixed('kernelsize2_2', 0)
        hp.Fixed('strides2', '')
    elif args.stacks==1:
        hp.Fixed('filters1_0', 0)
        hp.Fixed('filters1_1', 0)
        hp.Fixed('filters2_0', 0)
        hp.Fixed('filters2_1', 0)
        hp.Fixed('kernelsize1_0', 0)
        hp.Fixed('kernelsize1_1', 0)
        hp.Fixed('kernelsize1_2', 0)
        hp.Fixed('kernelsize2_0', 0)
        hp.Fixed('kernelsize2_1', 0)
        hp.Fixed('kernelsize2_2', 0)
        hp.Fixed('strides1', '')
        hp.Fixed('strides2', '')

    tuner = tunerClass(
        build_model,
        objective='val_accuracy',
        max_trials=args.max_trials,
        project_name=args.project_dir,
        hyperparameters=hp,
        overwrite=True)

    datagen.fit(X_train)

    print(tuner.search_space_summary())

    from tensorflow.keras.callbacks import LearningRateScheduler
    lr_schedule_func = get_lr_schedule_func(0.001, 0.99)

    callbacks = [LearningRateScheduler(lr_schedule_func, verbose=1)]

    tuner.search(datagen.flow(X_train, y_train, batch_size=32),                
                 epochs=10,
                 validation_data=(X_test, y_test),
                 callbacks=callbacks,
                 verbose=1
             )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tuner', choices=['RandomSearch','BayesianOptimization'], default = "RandomSearch", help="specify tuner")
    parser.add_argument('-p', '--project-dir', type=str, default = 'rs_resnet_v1_eembc', help = 'specify project dir')
    parser.add_argument('-m', '--max-trials', type=int, default = 100, help = 'specify max trials')
    parser.add_argument('-s', '--stacks', type=int, default = 3, help = 'specify number of stacks')

    args = parser.parse_args()

    main(args)
