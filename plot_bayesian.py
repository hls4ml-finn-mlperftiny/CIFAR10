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
import numpy as np
from bayesian import build_model
from tensorflow.keras.datasets import cifar10

def main(args):

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    num_classes = 10

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    results = {'filters0': [], 'filters1': [], 'filters2': [], 
               'kernelsize0': [], 'kernelsize1': [], 
               'strides0': [], 'strides1': [],
               'val_acc': [], 'flops': [], 'stacks': []}


    tuner_names = args.tuner.split(',')
    project_dirs = args.project_dir.split(',')
    
    tuners = []
    for tuner_name, project_dir in zip(tuner_names, project_dirs):
        tunerClass = getattr(kerastuner.tuners,tuner_name)
        tuner = tunerClass(
            build_model,
            objective='val_accuracy',
            max_trials=args.max_trials,
            project_name=project_dir,
            overwrite=False)
        tuners.append(tuner)

    import kerop
    for tuner in tuners:
        for trial, model, hp in zip(tuner.oracle.get_best_trials(num_trials=args.max_trials), 
                                    tuner.get_best_models(num_models=args.max_trials), 
                                    tuner.get_best_hyperparameters(num_trials=args.max_trials)):
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
            result['stacks'] = 1 + 1*(result['filters1']!=0) + 1*(result['filters2']!=0) 
            for key in result:
                results[key].append(result[key])
    # change to numpy array for easier indexing
    for key in result:
        results[key] = np.array(results[key])

    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)

    plt.figure()
        
    cmap = np.array(['white', 'blue', 'orange', 'green'])
    mask = (results['stacks']==3)
    if np.sum(mask)>0:
        plt.scatter(results['flops'][mask], results['val_acc'][mask], c=cmap[results['stacks'][mask]], label='3')
    mask = (results['stacks']==2)
    if np.sum(mask)>0:
        plt.scatter(results['flops'][mask], results['val_acc'][mask], c=cmap[results['stacks'][mask]], label='2')
    mask = (results['stacks']==1)
    if np.sum(mask)>0:
        plt.scatter(results['flops'][mask], results['val_acc'][mask], c=cmap[results['stacks'][mask]], label='1')
    plt.xlabel('FLOPs')
    plt.ylabel('Test accuracy')
    plt.legend(title='Stacks')
    plt.savefig('flops_val_acc.png')
    plt.savefig('flops_val_acc.pdf')
    plt.semilogx()
    plt.savefig('flops_val_acc_logx.png')
    plt.savefig('flops_val_acc_logx.pdf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tuner', type=str, default = "RandomSearch", help="specify tuner(s): multiple ones separated by commas")
    parser.add_argument('-p', '--project-dir', type=str, default = 'rs_resnet_v1_eembc', help = 'specify project dir(s): multiple ones separated by commas')
    parser.add_argument('-m', '--max-trials', type=int, default = 100, help = 'specify max trials')
    parser.add_argument('-s', '--stacks', type=int, default = 3, help = 'specify number of stacks')

    args = parser.parse_args()

    main(args)
