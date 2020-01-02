import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, Convolution2D, MaxPooling2D, BatchNormalization
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time


#unique name for model that has been trained
NAME = "CNN-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))


pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

# model = Sequential()

# model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

# model.add(Dense(64))

# model.add(Dense(1))
# model.add(Activation('softmax'))

# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

nb_filters = 64
pool_size = (2, 2)
kernel_size = (5, 5)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size, padding='valid', input_shape=(80, 80, 3)))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filters, kernel_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Dropout(0.25))

model.add(Flatten()) # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(  loss='sparse_categorical_crossentropy', 
                optimizer='adadelta',
                metrics=['accuracy'])


y = np.array(y)
model.fit(X, y, batch_size=64, epochs=10, validation_split=0.3, callbacks=[tensorboard])
