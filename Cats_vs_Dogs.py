#!/usr/bin/env python

import os
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# all directories
TRAIN_DIR = 'extracted/cats_and_dogs_filtered/train'
VAL_DIR = 'extracted/cats_and_dogs_filtered/validation'
TRAIN_DIR_CATS = 'extracted/cats_and_dogs_filtered/train/cats'
TRAIN_DIR_DOGS = 'extracted/cats_and_dogs_filtered/train/dogs'
VAL_DIR_CATS = 'extracted/cats_and_dogs_filtered/validation/cats'
VAL_DIR_DOGS = 'extracted/cats_and_dogs_filtered/validation/dogs'

# getting images squared
imageSize = 50

# alpha
lr = 0.001

# momentum
beta_1 = 0.9

# mini-batch size
BATCH_SIZE = 32

# number of epochs
epochs = 10

# total training data and val data
totalTrain = len(os.listdir(TRAIN_DIR_CATS))
totalTrain += len(os.listdir(TRAIN_DIR_DOGS))

totalVal = len(os.listdir(VAL_DIR_CATS))
totalVal += len(os.listdir(VAL_DIR_DOGS))

# for data augmentation since we have a small training set
trainImageGenerator = ImageDataGenerator(
                                        rescale = 1./255.,  # normalization
                                        rotation_range = 40,  # range of rotation
                                        width_shift_range = 0.2,
                                        height_shift_range = 0.2,
                                        shear_range = 0.2,
                                        zoom_range = 0.2,
                                        horizontal_flip = True,  # fliiping images horizontally
                                        fill_mode = 'nearest'  # any pixel gaps will be filled with the 'nearest' pizel
                                        )

validationImageGenerator = ImageDataGenerator(
                                        rescale = 1./255.
                                        )

# declaring training data
trainingData = trainImageGenerator.flow_from_directory(
                                        batch_size = BATCH_SIZE,
                                        directory = TRAIN_DIR,
                                        shuffle = True,
                                        target_size = (imageSize, imageSize),
                                        class_mode = 'binary'   # only cats or dogs, binary classification
                                        )

validationData = validationImageGenerator.flow_from_directory(
                                        batch_size = BATCH_SIZE,
                                        directory = VAL_DIR,
                                        shuffle = False,
                                        target_size = (imageSize, imageSize),
                                        class_mode = 'binary'
                                        )

# printing classes
print(trainingData.class_indices)

# clearing previous models
K.backend.clear_session()

# defining the model
model = K.models.Sequential([
    Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = (imageSize, imageSize, 3)), # 3 for RGB
    MaxPooling2D(pool_size = (2, 2)),

    Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'),
    MaxPooling2D(pool_size = (2, 2)),
    BatchNormalization(),

    Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'),
    MaxPooling2D(pool_size = (2, 2)),
    BatchNormalization(),

    Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'),
    MaxPooling2D(pool_size = (2, 2)),
    BatchNormalization(),

    Dropout(rate = 0.5),        # regularization to reduce variance

    Flatten(),
    Dense(units = 256, activation = 'relu'),

    Dense(units = 2, activation = 'softmax')

], name = 'cats-vs-dogs')

# compiling the model
model.compile (
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
    )

# summary of the model
model.summary()

# fit is overloaded with fit_generator in TensorFlow 2.2+, hence, using fit
model.fit(
    trainingData,
    steps_per_epoch = int(np.ceil(totalTrain / float(BATCH_SIZE))),  # steps per epoch is defined as total by batch size
    epochs = epochs,
    validation_data = validationData,
    validation_steps = int(np.ceil(totalTrain / float(BATCH_SIZE)))
    )

model.evaluate(validationData, batch_size = BATCH_SIZE)

# saving json format of model
modelJSON = model.to_json()
with open('cats_vs_dogs.json', 'w') as file:
    file.write(modelJSON)

# saving weights of model
model.save_weights("cats_vs_dogs_weights.h5")

#printing confirmation
print("Saved model to disk")

# loading model
jsonFile = open('cats_vs_dogs.json', 'r')
loadedJSONModel = jsonFile.read()
jsonFile.close()
loadedModel = K.models.model_from_json(loadedJSONModel)

# loading weights using this model
loadedModel.load_weights('cats_vs_dogs_weights.h5')

# compiling the loaded model
loadedModel.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

# evaluating the loaded model
loadedModel.evaluate(validationData, batch_size = BATCH_SIZE)
