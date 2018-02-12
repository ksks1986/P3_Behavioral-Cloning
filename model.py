import numpy as np
from PIL import Image

from keras.models import load_model
import h5py
from keras import __version__ as keras_version


import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Activation
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator    
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Cropping2D
import matplotlib.pyplot as plt

import os
import csv

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # get first row as header.
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print(keras_version)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32, mode=0):
    #mode=0:train, 1:validation
    correction = 0.3 # this is a parameter to tune
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name         = './IMG/'+batch_sample[0].split('\\')[-1]
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])

                #when training, use the left and right cameras and flip image.
                if mode == 0:
                    name         = './IMG/'+batch_sample[1].split('\\')[-1]
                    left_image   = cv2.imread(name)
                    left_image   = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                    name         = './IMG/'+batch_sample[2].split('\\')[-1]
                    right_image  = cv2.imread(name)
                    right_image  = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

                    left_angle   = center_angle + correction
                    right_angle  = center_angle - correction

                    center_image_flip = np.fliplr(center_image)
                    left_image_flip   = np.fliplr(left_image)
                    right_image_flip  = np.fliplr(right_image)

                    center_angle_flip = -center_angle
                    left_angle_flip   = -left_angle
                    right_angle_flip  = -right_angle

                    images = images + [left_image, right_image, center_image_flip, left_image_flip, right_image_flip]
                    angles = angles + [left_angle, right_angle, center_angle_flip, left_angle_flip, right_angle_flip]

                images = images + [center_image]
                angles = angles + [center_angle]

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32, mode=0)
validation_generator = generator(validation_samples, batch_size=32, mode=1)

row, col, ch = 160, 320, 3  # image format  

# Preprocess incoming data, centered around zero with small standard deviation 

inputs = Input(shape=(row, col, ch))
x = Lambda(lambda x: x/255.0 - 0.5,
       input_shape=(row, col, ch),
       output_shape=(row, col, ch))(inputs)

#(160,320,3) ---> (90,320,3)
x = Cropping2D(cropping=((50,20), (0,0)), input_shape=(row, col, ch))(x)

#(90,320,3) ---> (84,314,16)
x = Conv2D(16, 7, 7, activation='relu', init='he_normal')(x)

#(84,314,16) ---> (42,157,16)
x = BatchNormalization()(x)    
x = MaxPooling2D(pool_size=(2, 2))(x)

#(42,157,16) ---> (36,151,16)
x = BatchNormalization()(x)    
x = Conv2D(16, 7, 7, activation='relu', init='he_normal')(x)

#(36,151,6) ---> (18,75,16)
x = BatchNormalization()(x)    
x = MaxPooling2D(pool_size=(2, 2))(x)

#(18,75,16) ---> (12, 69, 16)
x = BatchNormalization()(x)    
x = Conv2D(16, 7, 7, activation='relu', init='he_normal')(x)

#(12,69,16) ---> (4, 61, 32)
x = BatchNormalization()(x)    
x = Conv2D(32, 9, 9, activation='relu', init='he_normal')(x)

#(4,61,32) ---> (2,30,32)
x = BatchNormalization()(x)    
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(32, init='he_normal', activation='relu')(x)
predictions = Dense(1, init='he_normal', activation='linear')(x)

model = Model(input=inputs, output=predictions)

model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*6,
                    validation_data=validation_generator,  nb_val_samples=len(validation_samples),
                    nb_epoch=10, verbose=1)


model.save('./model.h5')

K.clear_session()


