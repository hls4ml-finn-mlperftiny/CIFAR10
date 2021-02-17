import os
import glob
import sys
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score
from resnet_v1_eembc import resnet_v1_eembc
import yaml
import csv
import setGPU
import kerop
from train import get_lr_schedule_func
import kerastuner
from kerastuner.tuners import BayesianOptimization, Hyperband, RandomSearch

# define cnn model
def build_model(hp):
    hp_filters0 = hp.Choice('filters0', [2, 4, 8, 16])
    hp_filters1 = hp.Choice('filters1', [4, 8, 16, 32])
    hp_filters2 = hp.Choice('filters2', [8, 16, 32, 64])
    hp_kernelsize0 = hp.Choice('kernelsize0', [1, 2, 3])
    hp_kernelsize1 = hp.Choice('kernelsize1', [1, 2, 3])
    hp_strides0 = hp.Choice('strides0', [1, 4])
    hp_strides1 = hp.Choice('strides1', [2, 3, 4])
    
    model = resnet_v1_eembc(input_shape=[32, 32, 3], num_classes=10,
                            num_filters=[hp_filters0, hp_filters0, hp_filters1],
                            kernel_sizes=[hp_kernelsize0, hp_kernelsize1],
                            strides=[hp_strides0, hp_strides1], l1p=0, l2p=1e-4)
    # compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

from tensorflow.keras.datasets import cifar10

def main(args):

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    num_classes = 10

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    tuner = BayesianOptimization(
        build_model,
        objective='val_accuracy',
        max_trials=100,
        project_name='bo_resnet_v1_eembc_10epoch_100maxtrials_lrdecay',
        overwrite=False)

    results = {'filters0': [], 'filters1': [], 'filters2': [], 
               'kernelsize0': [], 'kernelsize1': [], 
               'strides0': [], 'strides1': [],
               'val_acc': [], 'flops': []}

    import kerop
    num = 100
    for trial, model, hp in zip(tuner.oracle.get_best_trials(num_trials=num), 
                                tuner.get_best_models(num_models=num), 
                                tuner.get_best_hyperparameters(num_trials=num)):
        metrics_tracker = trial.metrics # type: MetricsTracker
        metric_histories = metrics_tracker.metrics # type: Dict[str, MetricHistory]
        val_accuracy_hist = metric_histories['val_accuracy'] # type: MetricHistory
        val_acc = val_accuracy_hist.get_history()[0].value[0]
        #val_acc = model.evaluate(X_test, y_test)[1] # save time by skipping this
        layer_name, layer_flops, inshape, weights = kerop.profile(model)
        total_flop = 0
        for name, flop, shape in zip(layer_name, layer_flops, inshape):
            total_flop += flop

        result = hp.values
        result['val_acc'] = val_acc
        result['flops'] = total_flop
        print(result)
        for key in result:
            results[key].append(result[key])

    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)

    plt.figure()
    plt.scatter(results['flops'], results['val_acc'])
    plt.xlabel('FLOPs')
    plt.ylabel('Test accuracy')
    plt.savefig('flops_val_acc.png')
    plt.savefig('flops_val_acc.pdf')
    plt.semilogx()
    plt.savefig('flops_val_acc_logx.png')
    plt.savefig('flops_val_acc_logx.pdf')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main(args)
