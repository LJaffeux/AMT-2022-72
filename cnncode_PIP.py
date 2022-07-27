#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 10:20:15 2021
training/tuning of CNN classification using keras-tuner 
@author: jaffeux
"""
import numpy as np
import scipy.io
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import h5py
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from kerastuner.tuners import RandomSearch
from kerastuner import HyperModel
from keras import callbacks
from kerastuner.tuners import RandomSearch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from time import sleep
from tqdm import tqdm
from os import walk
import pathlib
from pathlib import Path
import tensorflow as tf
from keras.preprocessing import image
import imageio
import pandas as pd
datafolder = os.getcwd()+'/PIP_training_data'


img_generator = keras.preprocessing.image_dataset_from_directory(
    datafolder,
    labels='inferred',
    class_names=['Compact Particles','Fragile Aggregates','Columns and Needles',
                 'Hexagonal Planar Crystals','Rimed Aggregates','Combinations of Bullets and Columns' ],
    batch_size=16,
    color_mode='grayscale',
    image_size=(110,110),
    validation_split=0.2,
    seed=3,
    subset="training",
)

data_aug=keras.Sequential([
                           keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
])
earlystop=callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=5, verbose=2, mode="auto", baseline=None, restore_best_weights=True)

#This function updates the learning rate every X epochs
def scheduler(epochx, ler):
  X=7
  if epochx % 5 == 0:
    return ler*0.9
  else:
    return ler

callback = keras.callbacks.LearningRateScheduler(scheduler)


class CNNHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = keras.Sequential()
        model.add(
            data_aug
        )
        model.add(
            Conv2D(
                filters=32,
                kernel_size=3,
                activation='relu',
                input_shape=self.input_shape
            )
        )
        model.add(MaxPooling2D(pool_size=2))
        model.add(
            Dropout(rate=hp.Float(
                'dropout_1',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            ))
        )
        model.add(
            Conv2D(
                filters=64,
                activation='relu',
                kernel_size=3
            )
        )
        model.add(MaxPooling2D(pool_size=2))
        model.add(
            Dropout(rate=hp.Float(
                'dropout_2',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            ))
        )
        model.add(
            Conv2D(
                filters=128,
                activation='relu',
                kernel_size=3
            )
        )
        model.add(MaxPooling2D(pool_size=2))
        model.add(
            Dropout(rate=hp.Float(
                'dropout_3',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            ))
        )
        model.add(
            Conv2D(
                filters=256,
                activation='relu',
                kernel_size=3
            )
        )
        model.add(MaxPooling2D(pool_size=2))
        model.add(
            Dropout(rate=hp.Float(
                'dropout_4',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            ))
        )
        model.add(
            Conv2D(
                filters=512,
                activation='relu',
                kernel_size=3
            )
        )
        model.add(MaxPooling2D(pool_size=2))
        model.add(
            Dropout(rate=hp.Float(
                'dropout_5',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            ))
        )
        model.add(Flatten())
        model.add(
            Dense(
                units=hp.Int(
                    'units',
                    min_value=32,
                    max_value=512,
                    step=32,
                    default=128
                ),
                activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu'
                )
            )
        )
        model.add(
            Dropout(
                rate=hp.Float(
                    'dropout_6',
                    min_value=0.0,
                    max_value=0.5,
                    default=0.25,
                    step=0.05
                )
            )
        )
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3
                )
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model



NUM_CLASSES = 6  # 
INPUT_SHAPE = (110, 110,1)  # 
SEED=9
MAX_TRIALS=10
EXECUTION_PER_TRIAL=1

hypermodel = CNNHyperModel(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

tuner = RandomSearch(
    hypermodel,
    objective='accuracy',
    seed=SEED,
    max_trials=MAX_TRIALS,
    executions_per_trial=EXECUTION_PER_TRIAL,
    directory='random_search',
    project_name='ice_crystal'
)
tuner.search_space_summary()
tuner.search(img_generator,epochs=40,callbacks=[earlystop,callback],verbose=1)

