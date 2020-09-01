from math import ceil
from keras.models import Model
import matplotlib.pyplot as plt
import csv
import cv2
import numpy as np
import os
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Cropping2D, Flatten, Dropout, Dense, Lambda, Conv2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import random

def plot_model(model, train_generator, train_samples, validation_generator, validation_samples, nbepochs):

    history_object = model.fit_generator(train_generator, validation_data = 
        validation_generator,
        nb_val_samples = len(validation_samples), 
        nb_epoch=nbepochs, verbose=1)
    
    ### print the keys contained in the history object
    print(history_object.history.keys())
    
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

def generator(samples, batch_size=32):
    num_samples = len(samples)
    base_path = './data/'
    correction_factor = [0.25, 0, -0.25] # Read http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf    
    while 1: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for line in batch_samples:
                for i in range(3):
                    source_path = line[i]
                    file_name = source_path.split('\\')[-1]
                    current_path = base_path + file_name
                    image = cv2.imread(current_path)
                    #image[:,:,0] = cv2.resize(image.squeeze(), (320,160))
                    measurement = float(line[3]) + correction_factor[i]
                    
                    images.append(image)
                    measurements.append(measurement)
                    if np.random.uniform()>0.5:
                        image_flipped = np.fliplr(image)
                        measurement_flipped = -measurement

                        images.append(image_flipped)
                        measurements.append(measurement_flipped)
                    if np.random.uniform()>0.5:
                        pix2angle = -0.05 #Opposed direction
                        latShift = random.randint(-5,5) 
                        M = np.float32([[1,0,latShift],[0,1,0]])
                        img_translated = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
                        images.append(img_translated)
                        measurements.append(measurement)



            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)



nbepoch = 5
batch_size=32
ch, row, col = 3, 160, 320  # Trimmed image format

# compile and train the model using the generator function
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)



model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (row, col, ch)))
model.add(Cropping2D(cropping = ((60,25), (0, 0)))) # Crops 70 fom the tp, 5 from the bottom, 0 from the left, 0 from the right.
model.add(Conv2D(filters=24,kernel_size=(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(filters=36,kernel_size=(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(filters=48,kernel_size=(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation="relu"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1164))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

# NVidia network: https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/6b6c37bc-13a5-47c7-88ed-eb1fce9789a0/lessons/3fc8dd70-23b3-4f49-86eb-a8707f71f8dd/concepts/7f68e171-cf87-40d2-adeb-61ae99fe56f5

model.compile(loss='mse', optimizer='adam')

#plot_model(model, train_generator, train_samples, validation_generator, validation_samples, nbepoch)

model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=ceil(len(validation_samples)/batch_size), epochs=nbepoch, verbose=1)
            
model.save('model.h5')





