import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, Convolution2D, MaxPooling2D, BatchNormalization
import numpy as np
from loadData import create_training_data
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

from sklearn.metrics import classification_report

#unique name for model that has been trained
NAME = "CNN-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))

pickle_in = open("X_train.pickle","rb")
X_train = pickle.load(pickle_in)

pickle_in = open("y_train.pickle","rb")
y_train = pickle.load(pickle_in)

pickle_in = open("X_test.pickle","rb")
X_test = pickle.load(pickle_in)

pickle_in = open("y_test.pickle","rb")
y_test = pickle.load(pickle_in)

X_train = X_train/255.0

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

def build_model_1():
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
    return model

model_1 = build_model_1()

y_train = np.array(y_train)
y_test = np.array(y_test)

X_test = np.array(X_test)

# train the model 
model_1.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.15, callbacks=[tensorboard])

# test the model
score = model_1.evaluate(X_test, y_test)
print("\nTest accuracy: %0.05f" % score[1], "\n")

X_test = tf.convert_to_tensor(X_test,dtype=tf.int32)
y_test = tf.convert_to_tensor(y_test,dtype=tf.int32)

# needed conversion in order to work bellow
X_test = np.array(X_test).astype(np.float32)

# classification report
y_pred = model_1.predict(X_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))