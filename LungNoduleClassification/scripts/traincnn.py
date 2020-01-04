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
from collections import Counter 

from sklearn.metrics import classification_report

#unique name for model that has been trained
NAME = "CNN-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))

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

def create_model(layer_size, dense_layers, conv_layers):
    pool_size = (2, 2)
    kernel_size = (5, 5)

    model = Sequential()

    model.add(Convolution2D(layer_size, kernel_size, padding='valid', input_shape=(80, 80, 3)))
    model.add(Activation('relu'))

    for l in range(conv_layers - 1):
        model.add(Convolution2D(layer_size, kernel_size))
        model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Dropout(0.25))

    model.add(Flatten()) # this converts our 3D feature maps to 1D feature vectors
    
    for l in range(dense_layers):    
        model.add(Dense(layer_size))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

    model.add(Dense(3))
    model.add(Activation('softmax')) # or sigmoid?

    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return  {cls: float(majority/count) for cls, count in counter.items()}

def performance_stats(model):
    pickle_in = open("X_train.pickle","rb")
    X_train = pickle.load(pickle_in)

    pickle_in = open("y_train.pickle","rb")
    y_train = pickle.load(pickle_in)

    pickle_in = open("X_test.pickle","rb")
    X_test = pickle.load(pickle_in)

    pickle_in = open("y_test.pickle","rb")
    y_test = pickle.load(pickle_in)

    X_train = X_train/255.0
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    X_test = np.array(X_test)

    #class_weight = {2: 1., 1: 10., 2: 2.}

    class_weights = get_class_weights(y_train)

    print(class_weights)

    # train the model 
    model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.15, callbacks=[tensorboard])
    #model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.15, class_weight=class_weights)

    # test the model
    score = model.evaluate(X_test, y_test)
    print("\nTest accuracy: %0.05f" % score[1], "\n")

    X_test = tf.convert_to_tensor(X_test,dtype=tf.int32)
    y_test = tf.convert_to_tensor(y_test,dtype=tf.int32)

    # needed conversion in order to work bellow
    X_test = np.array(X_test).astype(np.float32)

    # classification report
    y_pred = model.predict(X_test, batch_size=64, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred_bool))

#build model
model_1 = create_model(128, 2, 1)

#check performance
performance_stats(model_1)

