# -*- coding: utf-8 -*-
"""
**********Facial Recognition VGG16**********
**********@author: gulsen************
"""
""" *********************************************************************** """
""" 
Libraries
"""

import numpy as np
from tqdm import tqdm #allows for the generation of progress bars
import cv2 #Open Source Computer Vision Library
import os
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.applications.vgg19 import VGG19

init_notebook_mode(connected=True)
RANDOM_SEED=123

""" *********************************************************************** """
"""
Data Import
"""

TRAIN_DIR=(r'C:/Users/LENOVO/Desktop/data_science/Facial_Recognition_Data/Training/')
TEST_DIR=(r'C:/Users/LENOVO/Desktop/data_science/Facial_Recognition_Data/Testing/')

""" *********************************************************************** """

"""
Data Preparation
"""

def load_data(dir_path, IMG_SIZE):
    
    X=[]
    y=[]
    i=0
    labels=dict() #associative array
    for path in tqdm(sorted(os.listdir(dir_path))): #os.listdir:get the list of all files and directories in the specified directory
        if not path.startswith('.'):
            labels[i]=path
            for file in os.listdir(dir_path + path):
                if not file.startswith('.'):
                   img=cv2.imread(dir_path + path + '/' + file)
                   img=img.astype('float32') / 255
                   resized=cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA) #interpolation: estimate unknown data points between two known data points
                   X.append(resized)
                   y.append(i)
            i+=1
    X=np.array(X)
    y=np.array(y)
    print(f'{len(X)} images laded from {dir_path} directory.')
    return X,y, labels

IMG_SIZE=(48,48)

X_train, y_train, train_labels = load_data(TRAIN_DIR, IMG_SIZE)

train_labels

X_test, y_test, test_labels = load_data(TEST_DIR, IMG_SIZE)

def plot_samples(X, y, labels_dict, n=50):
    
    for index in range (len(labels_dict)):
        imgs= X[np.argwhere(y==index)] [:n] #argwhere: Indices of elements that are non-zero
        j=10 #piece
        i=int(n/j)
        
        plt.figure(figsize=(10,3))
        c=1 #cols
        for img in imgs:
            plt.subplot(i,j,c)
            plt.imshow(img[0])
            
            plt.xticks([])
            plt.yticks([])
            c+= 1
        plt.suptitle(labels_dict[index])
        plt.show()
        
plot_samples(X_train, y_train, train_labels, 10)

""" *********************************************************************** """

"""
Data PROCESSİNG
"""

Y_train = to_categorical(y_train, num_classes=6) #Quality of use data and appropriate in-formula for CNN
Y_train.shape

Y_test = to_categorical(y_test, num_classes=6)
Y_test.shape

# """ *********************************************************************** """

# """
# MODEL BUILDING
# """
       
# base_model= VGG19(
#         weights=None, #connection managements between two basic units within a neural network
#         include_top=False, #in order to exclude the model's fully-connected layers
#         input_shape=IMG_SIZE + (3,)
#     )

# base_model.summary()

# NUM_CLASSES= 6

# model=Sequential()
# model.add(base_model)
# model.add(Flatten())
# model.add(Dense(1000, activation="relu"))
# model.add(Dropout(0.1)) #simple and powerful regularization technique
# model.add(Dense(NUM_CLASSES, activation="softmax"))

# def deep_model(model, X_train, Y_train, epochs, batch_size): #batch:number of subtests
    
#     model.compile(
#     loss='binary_crossentropy',
#     optimizer= RMSprop(learning_rate=1e-4),
#     metrics=['accuracy'])
    
#     history=model.fit(X_train,
#                       Y_train,
#                       epochs=epochs,
#                       batch_size=batch_size)
#     return history

# epochs=1
# batch_size=128

# history=deep_model(model, X_train, Y_train, epochs, batch_size)

""" *********************************************************************** """

"""
CONFUSION MATRIX
"""

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
   plt.figure(figsize=(6,6))
   plt.imshow(cm, interpolation='nearest', cmap=cmap)
   plt.title(title)
   plt.colorbar()
   ticks_marks=np.arrange(len(classes))
   plt.xticks(ticks_marks, classes, rotation=90)
   plt.yticks(ticks_marks, classes)
   
   if normalize:
      cm=cm.astype('float') / cm.sum(axis=1) [:, np.newaxis]
    
   thresh=cm.max() / 2
   cm=np.round(cm,2)
   for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
       plt.text(j,i, cm [i,j],
                horizontalalignment="center",
                color="white" if cm[i,j] > thresh else "black")
   plt.tight_layout()
   plt.ylabel('True label')
   plt.xlabel('Predicted label')
   plt.show()
   
predictions=model.predict(X_test)
y_pred=[np.argmax(probas) for probas in predictions]

accuracy=accuracy_store(y_test, y_pred)
print('Test Accuracy = %.2f' % accuracy)

confusion_mtx=confusion_matrix(y_test, y_pred)
cm=plot_confusion_matrix(confysion_mtx, classes=list(test_labels.items))
          

