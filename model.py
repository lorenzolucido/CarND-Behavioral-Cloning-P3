
# ------ Imports and helper functions  -------

import csv
import cv2
import numpy as np
import sklearn
import random


def get_driving_log(path):
    lines = []
    with open(path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
            
    return lines


def generator(path, samples, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for image_nb in [0, 1, 2]:
                    name = path + 'IMG/'+ batch_sample[image_nb].split('/')[-1]
                    image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                    shift_dict = {0: 0, 1: 0.3, 2: -0.3} 
                    angle = float(batch_sample[3]) + shift_dict.get(image_nb, "error")
    
                    images.append(image)            
                    angles.append(angle)
    
                    images.append(np.fliplr(image))
                    angles.append(-angle)            

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train)

# ------ Training/Validation data loading -------

path = './data/Lolo_data/'

samples = get_driving_log(path)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(path, train_samples, batch_size=32)
validation_generator = generator(path, validation_samples, batch_size=32)


# ------ Model Definition -------

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.backend import tf

def resize_im(x):
    from keras.backend import tf
    return tf.image.resize_images(x, (80, 160))

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 127.5 - 1))
model.add(Lambda(resize_im))

# -- Convolutions
model.add(Convolution2D(16,3,3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32,3,3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64,3,3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128,3,3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# -- Flattening
model.add(Flatten())

# -- Dense layers
model.add(Dense(1000, activation='relu'))
model.add(Dropout(p=0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(p=0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(p=0.5))

# -- Output
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')


# ------ Model Training and Saving -------

from keras.models import Model
import matplotlib.pyplot as plt

history_object = model.fit_generator(train_generator, 
                                     samples_per_epoch = len(train_samples)*6, 
                                     validation_data = validation_generator,    
                                     nb_val_samples = len(validation_samples)*6,     
                                     nb_epoch=5, 
                                     verbose=1)

model.save('model.h5')