import tensorflow as tf
import pickle
import numpy as np
import cv2

from sklearn.metrics import classification_report

def prepare(filepath):
    IMG_SIZE = 80
    img_array = cv2.imread(filepath)
    img_array = img_array/255.0
    return img_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

def predict_values(modelName, pathImage):
    model = tf.keras.models.load_model(modelName)
    prediction = model.predict(prepare(pathImage))

    return prediction

def main():
    pathImage = "../Dataset/images/PartSolid/LNDb-0015_finding1.png"
    pred = predict_values("qq.model", pathImage)
    print(pred)

main()