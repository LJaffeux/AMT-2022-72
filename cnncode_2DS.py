#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 10:20:15 2021
training/tuning of CNN classification using keras-tuner 
@author: jaffeux
"""

 
 #train set split en train + validation
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
datafolder=os.getcwd()+'/DS_training_data'
img_generator = keras.preprocessing.image_dataset_from_directory(
    datafolder,
    batch_size=32,
    image_size=(200,200)
)
imb=[]
for name in tqdm(img_generator.file_paths):
    sleep(0.0001)
    im=imageio.imread(name)
    ima= np.reshape(im,(np.shape(im)[0],np.shape(im)[1],1))
    imagex = tf.image.resize_with_crop_or_pad(ima, 200, 200)
    imb.append(imagex)
    imageio.imwrite(name,imagex)

img_generator = keras.preprocessing.image_dataset_from_directory(
    datafolder,
    labels='inferred',
    class_names=['Compact Particles','Fragile Aggregates','Columns and Needles','Hexagonal Planar Crystals',
                 'Combinations of Bullets and Columns','Complex Assemblages of planes, columns, dendrites' ,
                 'Capped Columns', 'Water Droplets','Diffracted'],
    batch_size=16,
    color_mode='grayscale',
    image_size=(200,200),
    validation_split=0.2,
    seed=1,
    subset="training"
)


data_aug=keras.Sequential([
                           layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
])
earlystop=callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=5, verbose=0, mode="auto", baseline=None, restore_best_weights=True)

#This function updates the learning rate every X epochs
def scheduler(epochx, ler):
  X=7
  if epochx % 5 == 0:
    return ler*0.95
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
                filters=64,
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
                filters=128,
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
                filters=256,
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
                filters=512,
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
                filters=1024,
                activation='relu',
                kernel_size=3
            )
        )       
        model.add(MaxPooling2D(pool_size=2))
        model.add(
            Dropout(rate=hp.Float(
                'dropout_7',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            ))
        )
        model.add(
            Conv2D(
                filters=2048,
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
                    min_value=64,
                    max_value=2048,
                    step=32,
                    default=256
                ),
                activation='relu'
                
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
INPUT_SHAPE = (200, 200,1)  # 
#INPUT_SHAPE = (110, 110,1)  # 
SEED = 10
MAX_TRIALS = 4
EXECUTION_PER_TRIAL = 2

hypermodel = CNNHyperModel(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

tuner = RandomSearch(
    hypermodel,
    objective='loss',
    seed=SEED,
    max_trials=MAX_TRIALS,
    executions_per_trial=EXECUTION_PER_TRIAL,
    directory='random_search_DS',
    project_name='ice_crystal'
)
tuner.search_space_summary()
tuner.search(img_generator,epochs=40,callbacks=[earlystop,callback],verbose=1)
