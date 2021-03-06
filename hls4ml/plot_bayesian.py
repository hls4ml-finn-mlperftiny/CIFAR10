import os
import glob
import sys
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score
import yaml
import csv
import setGPU
import kerop
from train import get_lr_schedule_func
import kerastuner
import numpy as np
from bayesian import build_model
from tensorflow.keras.datasets import cifar10
import pandas as pd


def main(args):

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    num_classes = 10

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    results = {'filters0_0': [], 'filters0_1': [],
               'filters1_0': [], 'filters1_1': [],
               'filters2_0': [], 'filters2_1': [],
               'kernelsize0_0': [], 'kernelsize0_1': [], 'kernelsize0_2': [],
               'kernelsize1_0': [], 'kernelsize1_1': [], 'kernelsize1_2': [],
               'kernelsize2_0': [], 'kernelsize2_1': [], 'kernelsize2_2': [],
               'strides0': [],
               'strides1': [],
               'strides2': [],
               'val_acc': [], 'flops': [], 'stacks': []}

    tuner_names = args.tuner.split(',')
    project_dirs = args.project_dir.split(',')

    tuners = []
    for tuner_name, project_dir in zip(tuner_names, project_dirs):
        tunerClass = getattr(kerastuner.tuners, tuner_name)
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
            metrics_tracker = trial.metrics  # type: MetricsTracker
            metric_histories = metrics_tracker.metrics  # type: Dict[str, MetricHistory]
            val_accuracy_hist = metric_histories['val_accuracy']  # type: MetricHistory
            val_acc = val_accuracy_hist.get_history()[0].value[0]
            # val_acc = model.evaluate(X_test, y_test)[1] # save time by skipping this
            layer_name, layer_flops, inshape, weights = kerop.profile(model)
            total_flop = 0
            for name, flop, shape in zip(layer_name, layer_flops, inshape):
                total_flop += flop

            result = hp.values
            result['val_acc'] = val_acc
            result['flops'] = total_flop
            result['stacks'] = 1 + 1*(result['filters1_0'] != 0) + 1*(result['filters2_0'] != 0)
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
    mask = (results['stacks'] == 3)
    if np.sum(mask) > 0:
        plt.scatter(results['flops'][mask], results['val_acc'][mask], c=cmap[results['stacks'][mask]], label='3')
    mask = (results['stacks'] == 2)
    if np.sum(mask) > 0:
        plt.scatter(results['flops'][mask], results['val_acc'][mask], c=cmap[results['stacks'][mask]], label='2')
    mask = (results['stacks'] == 1)
    if np.sum(mask) > 0:
        plt.scatter(results['flops'][mask], results['val_acc'][mask], c=cmap[results['stacks'][mask]], label='1')
    plt.xlabel('FLOPs')
    plt.ylabel('Test accuracy')
    plt.legend(title='Stacks')
    plt.savefig(os.path.join(project_dir, 'flops_val_acc.png'))
    plt.savefig(os.path.join(project_dir, 'flops_val_acc.pdf'))
    plt.semilogx()
    plt.savefig(os.path.join(project_dir, 'flops_val_acc_logx.png'))
    plt.savefig(os.path.join(project_dir, 'flops_val_acc_logx.pdf'))

    import pickle
    f = open(os.path.join(project_dir, "results.pkl"), "wb")
    pickle.dump(results, f)
    f.close()

    print("best models")
    df = pd.DataFrame.from_dict(results)
    #df['val_acc_over_log_flops'] = df['val_acc']/np.log10(df['flops'])
    # sort by val accuracy
    df.sort_values('val_acc', inplace=True, ascending=False)
    # drop duplicate hyperparameters (but keep highest accuracy)
    df.drop_duplicates(subset=['filters0_0', 'filters0_1',
                               'filters1_0', 'filters1_1',
                               'filters2_0', 'filters2_1',
                               'kernelsize0_0', 'kernelsize0_1', 'kernelsize0_2',
                               'kernelsize1_0', 'kernelsize1_1', 'kernelsize1_2',
                               'kernelsize2_0', 'kernelsize2_1', 'kernelsize2_2',
                               'strides0', 'strides1', 'strides2'],
                       inplace=True, keep='first')
    # print all models with < 1 MFLOPs
    # print(df[df['flops']<1e6].to_string())
    # print all models with < 20 MFLOPs
    print(df[df['flops'] < 2e7].to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tuner', type=str, default="RandomSearch", help="specify tuner(s): multiple ones separated by commas")
    parser.add_argument('-p', '--project-dir', type=str, default='rs_resnet_v1_eembc', help='specify project dir(s): multiple ones separated by commas')
    parser.add_argument('-m', '--max-trials', type=int, default=100, help='specify max trials')
    parser.add_argument('-s', '--stacks', type=int, default=3, help='specify number of stacks')

    args = parser.parse_args()

    main(args)
