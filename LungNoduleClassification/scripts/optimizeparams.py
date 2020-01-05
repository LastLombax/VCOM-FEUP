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
import utils


from sklearn.metrics import classification_report

pickle_in = open("X_train.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y_train.pickle","rb")
y = pickle.load(pickle_in)

pickle_in = open("X_test.pickle","rb")
X_test = pickle.load(pickle_in)

pickle_in = open("y_test.pickle","rb")
y_test = pickle.load(pickle_in)

X = X/255.0

X_test = np.array(X_test)
X_test = X_test/255.0

score_file = open("model_scores.txt", "a")

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
            
            model = utils.create_model(layer_size, dense_layer, conv_layer)

            y = np.array(y)
            model.fit(X, y, batch_size=64, epochs=10, validation_split=0.3, callbacks=[tensorboard])

            X_test = tf.convert_to_tensor(X_test,dtype=tf.int32)
            y_test = tf.convert_to_tensor(y_test,dtype=tf.int32)

            # needed conversion in order to work bellow
            X_test = np.array(X_test).astype(np.float32)

            # classification report
            y_pred = model.predict(X_test, batch_size=64, verbose=1)
            y_pred_bool = np.argmax(y_pred, axis=1)
            print(classification_report(y_test, y_pred_bool))
            
            score_file.write(NAME + "\n")
            score_file.write(classification_report(y_test, y_pred_bool) + "\r\n")

score_file.close()