import numpy as np
import cv2
import sys
import os
import sklearn
import csv
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Lambda
from keras.layers import Convolution2D, Dropout
from keras.layers import Flatten, Dense, AveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#read image paths and labels from logs
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#define generator to produce training data
def generator(samples, batch_size=32):
    num_samples = len(samples)
    angle_correction=0.3 #very important hp, tuned after many tests
    while 1: # Loop forever so the generator never terminates
        shuffle(samples) #randomly shuffle data to avoid getting stuck in local minima
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                cname = './data/IMG/'+batch_sample[0].split('/')[-1]
                lname = './data/IMG/'+batch_sample[1].split('/')[-1]
                rname = './data/IMG/'+batch_sample[2].split('/')[-1]
                #the images in my environment are stored at ./data/IMG/

                center_image = cv2.imread(cname,1)
                center_image = center_image[...,::-1] #convert from BGR to RGB
                center_angle = float(batch_sample[3])
                
                left_image = cv2.imread(lname)
                left_image = left_image[...,::-1] #convert from BGR to RGB
                left_angle = center_angle+angle_correction #add correction from center angle
                
                right_image = cv2.imread(rname)
                right_image = right_image[...,::-1] #convert from BGR to RGB
                right_angle= center_angle-angle_correction #subtract correction from center angle
                
                images.extend([center_image,left_image,right_image])
                angles.extend([center_angle,left_angle,right_angle])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            for b in range(3): #loop thrice as using left, right and center images leads to three times the data
                yield sklearn.utils.shuffle(X_train[b*batch_size:(b+1)*batch_size,...], y_train[b*batch_size:(b+1)*batch_size,...])



### Preparing training and validation datasets

batch_size=32
samples=samples[1:] #skip header
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size=batch_size) #generator for training data
validation_generator = generator(validation_samples, batch_size=batch_size) #generator for validation data


### Model architecture

model=Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)),input_shape=(160,320,3))) #Crop images to 90x320
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(90,320,3))) #normalize them

#add 16 conv filters with 2x2 strides. Output size= 45x160x16
model.add(Convolution2D(filters=16,kernel_size=5,strides=(2,2), padding='same',activation='relu'))

#Convolution layer with 32 filters. Output size= 23x80x32
model.add(Convolution2D(filters=32,kernel_size=5,strides=(2,2), padding='same',activation='relu'))

#Convolution layer with 64 filters. Output size= 12x40x64
model.add(Convolution2D(filters=64,kernel_size=5,strides=(2,2), padding='same',activation='relu'))

#Convolution layer with 128 filters. Output size= 6x20x128
model.add(Convolution2D(filters=128,kernel_size=5,strides=(2,2), padding='same',activation='relu'))

#Convolution layer with 256 filters. Output size= 3x10x256
model.add(Convolution2D(filters=256,kernel_size=3,strides=(2,2), padding='same',activation='relu'))

#Flatten tensor from 3D to 1D
model.add(Flatten())

#Dropout layer to reduce overfitting in dense layer
model.add(Dropout(0.3))

#Dense layer with 256 neurons
model.add(Dense(256, activation='relu'))

#Output of the model with tanh activation to make it in range [-1,1]
model.add(Dense(1,activation='tanh'))

#Use mean squared error as loss and use adam optimizer
model.compile(loss='mse', optimizer='adam')

#fit the model to the data using the generators defined above
model.fit_generator(train_generator, steps_per_epoch=np.ceil(len(train_samples)/batch_size),
        validation_data=validation_generator, validation_steps=np.ceil(len(validation_samples)/batch_size),
        epochs=8, verbose=1)

#save the model to an h5 file
model.save('model.h5')
