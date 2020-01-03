import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random

DATADIR = "../Dataset/images"

CATEGORIES = ["GGO", "PartSolid", "Solid"]

IMG_SIZE = 80

training_data = []

# sample_type : 0 for nothing, 1 for undersample and 2 for oversample
# sample_ratio: ignored if sample_type is 0. For over and under sample is the percentage of (over or under) sampling given the training_data
# test_ratio: percentage of test samples considering the training data 
def create_training_data(sample_type, sample_ratio, test_ratio):
    ggo = []
    partSolid = []
    solid = []
    for category in CATEGORIES:  # do ggo, ps and s

        path = os.path.join(DATADIR,category)  # create path to ggo, ps and s
        class_num = CATEGORIES.index(category)  # get the classification  (0, 1 or 2). 

        for img in tqdm(os.listdir(path)):  # iterate over each image per ggo, ps and s
            try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                #new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                if category == "GGO":
                    ggo.append([img_array, class_num])  # add this to our training_data
                elif category == "PartSolid":
                    partSolid.append([img_array, class_num])  # add this to our training_data
                elif category == "Solid":
                    solid.append([img_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))
    random.shuffle(ggo)
    random.shuffle(partSolid)
    random.shuffle(solid)

    training_data = ggo + partSolid + solid

    print(" (Before Sampling) Number of GGO images: ", len(ggo))
    print(" (Before Sampling) Number of PartSolid images: ", len(partSolid))
    print(" (Before Sampling) Number of Solid images: ", len(solid))

    print(" (Before Sampling) Total number of images: ", len(training_data), "\n")

    # undersample
    if (sample_type == 1):
        solid = random.sample(solid, int(sample_ratio * len(solid)))
    # oversample 
    elif (sample_type == 2):
        #it is being done......
        solid = solid

    training_data = ggo + partSolid + solid

    print(" (After Sampling) Number of GGO images: ", len(ggo))
    print(" (After Sampling) Number of PartSolid images: ", len(partSolid))
    print(" (After Sampling) Number of Solid images: ", len(solid))

    print(" (After Samples) Total number of images: ", len(training_data), "\n")

    random.shuffle(training_data)

    #for sample in training_data[:10]:
        #print(sample[1])

    X_train = []
    y_train = []

    X_test = []
    y_test = []

    nr_training_samples = len(training_data) - int(test_ratio * len(training_data))

    print(" Total number of test samples: ", int(len(training_data) - nr_training_samples), "\n")

    ## 80% for train
    for features,label in training_data[:nr_training_samples]:
        X_train.append(features)
        y_train.append(label)

    ## 20% for test
    for features,label in training_data[nr_training_samples:]:
        X_test.append(features)
        y_test.append(label)

    #1 because its made for grayscale
    #print(X_train[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3))

    X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

    import pickle

    pickle_out = open("X_train.pickle","wb")
    pickle.dump(X_train, pickle_out)
    pickle_out.close()

    pickle_out = open("y_train.pickle","wb")
    pickle.dump(y_train, pickle_out)
    pickle_out.close()

    pickle_out = open("X_test.pickle","wb")
    pickle.dump(X_test, pickle_out)
    pickle_out.close()

    pickle_out = open("y_test.pickle","wb")
    pickle.dump(y_test, pickle_out)
    pickle_out.close()

    pickle_in = open("X_train.pickle","rb")
    X_train = pickle.load(pickle_in)

    pickle_in = open("y_train.pickle","rb")
    y_train = pickle.load(pickle_in)

    pickle_in = open("X_test.pickle","rb")
    X_test = pickle.load(pickle_in)

    pickle_in = open("y_test.pickle","rb")
    y_test = pickle.load(pickle_in)

# (1) sample_type : 0 for nothing, 1 for undersample and 2 for oversample
# (2) sample_ratio: ignored if sample_type is 0. For over and under sample is the percentage of (over or under) sampling given the training_data
# (3) test_ratio: percentage of test samples considering the training data 
create_training_data(1, 0.08, 0.2)