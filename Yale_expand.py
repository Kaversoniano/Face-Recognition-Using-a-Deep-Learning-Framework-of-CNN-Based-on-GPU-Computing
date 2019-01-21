### Environment Preparation
import sys
sys.path.append('F:/Python/neophyte/venv')
sys.path.append('C:\\Users\\Kiano\\AppData\\Local\\Programs\\Python\\Python37-32\\Scripts')
sys.path.append('C:\\Users\\Kiano\\AppData\\Local\\Programs\\Python\\Python37-32\\Lib\\site-packages')
sys.path.append('C:\\Users\\Kiano\\AppData\\Roaming\\Python\\Python36\\Scripts')
sys.path.append('C:\\Users\\Kiano\\AppData\\Roaming\\Python\\Python36\\site-packages')

import training_expand
from training_expand import ImageTranslation, ImageRotation1, ImageRotation2

import NN_CNN
from NN_CNN import Network
from NN_CNN import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

import theano
import theano.tensor as T

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

theano.config.exception_verbosity='high'

def shared(data):
    """Place the data into shared variables.  This allows Theano to copy
       the data to the GPU, if one is available."""
    shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
    return shared_x, T.cast(shared_y, "int32")

### Data Read-in
# class names
individuals = ['yaleB0' + str(k) for k in range(1,10)]
individuals.extend(['yaleB' + str(k) for k in range(10,14)])
individuals.extend(['yaleB' + str(k) for k in range(15,40)])

# get data - images and labels
dataset = []
labels = []

for id in individuals:
    filename = []
    for files in os.walk('F:\\Python\\face\\CroppedYale\\' + id):
        filename.append(files)
    imgname = [filename[0][2][k] for k in range(2, len(filename[0][2])-1)]
    labels.extend([id for k in range(len(imgname))])
    for k in imgname:
        img = cv2.imread('F:\\Python\\face\\CroppedYale\\' + id + '\\' + k)
        img = (1/3*img[:, :, 0] + 1/3*img[:, :, 1] + 1/3*img[:, :, 2])/255
        img = img.flatten()
        dataset.append(img)

# convert labels(string) to labels(integer)
labels = pd.Series(labels)
labelsInt = []
for k in range(0,len(individuals)):
    single_labels = labels[labels==individuals[k]]
    single_labels = (single_labels==individuals[k])*k
    labelsInt.extend(list(single_labels))
labels = pd.Series(labelsInt)

# training set, validation set, test set
train_y = labels.sample(frac=0.7)
vt_y = labels.drop(train_y.index)
validation_y = vt_y.sample(frac=0.5)
test_y = vt_y.drop(validation_y.index)

dataset = np.array(dataset)
train_x = dataset[list(train_y.index)]
vt_x = dataset[list(vt_y.index)]
validation_x = dataset[list(validation_y.index)]
test_x = dataset[list(test_y.index)]

# format unification
train_y, validation_y, test_y = np.array(train_y), np.array(validation_y), np.array(test_y)

validation_data = (validation_x, validation_y)
test_data = (test_x, test_y)

# artificially expanding the training set
for k in range(train_x.shape[0]):
    # appending flattened image data to the end of train_x
    image = train_x[k].reshape((192,168))
    imageT = ImageTranslation(image,10,10).flatten() # translation with both horizontally and vertically ±10
    imageR1 = ImageRotation1(image,1,6).flatten() # counter-clockwise rotation with degree U(1,6)
    imageR2 = ImageRotation2(image,1,6).flatten() # clockwise rotation with degree U(1,6)
    train_x = np.append(train_x, [imageT,imageR1,imageR2], axis=0)

    # appending labels to the end of train_y
    flag = [train_y[k] for r in range(3)]
    train_y = np.append(train_y, flag)

training_data = (train_x, train_y) # size of 1690x4=6760, divisible by 10

# data preparation for GPU computing
training_data = shared(training_data)
validation_data = shared(validation_data)
test_data = shared(test_data)

### Building and Training Model
mini_batch_size = 10

net = Network([ConvPoolLayer(image_shape=(mini_batch_size, 1, 192, 168),
                             filter_shape=(20, 1, 31, 31),
                             poolsize=(2, 2),
                             activation_fn=NN_CNN.ReLU),
               FullyConnectedLayer(n_in=20*81*69, n_out=200, activation_fn=NN_CNN.ReLU, p_dropout=0.5),
               SoftmaxLayer(n_in=200, n_out=38)], mini_batch_size)

net.SGD(training_data, 50, mini_batch_size, 0.005, validation_data, test_data, lmbda = 0.5)

# best validation accuracy to date: 98.34% (47th epoch), its corresponding test accuracy: 98.34%