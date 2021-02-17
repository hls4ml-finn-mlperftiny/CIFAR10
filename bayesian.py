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
    
    # define data generator                                                                                                      
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )

    tuner = BayesianOptimization(
        build_model,
        objective='val_accuracy',
        max_trials=100,
        project_name='bo_resnet_v1_eembc_10epoch_100maxtrials_lrdecay',
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

    args = parser.parse_args()

    main(args)
