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
import traincnn


pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

dense_layers = [1, 2, 3]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

pool_size = (2, 2)
kernel_size = (5, 5)

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time())) #unique name for model that has been trained
            tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))
            
            model = traincnn.create_model(layer_size, dense_layer, conv_layer)

            y = np.array(y)
            model.fit(X, y, batch_size=64, epochs=10, validation_split=0.3, callbacks=[tensorboard])
