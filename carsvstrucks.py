#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


images = []
labels = []
IMG_SIZE=50
dir_car ='/Users/joe/Desktop/BU_2018_fall/EC601/miniProject2/Datasets/car'
dir_truck ='/Users/joe/Desktop/BU_2018_fall/EC601/miniProject2/Datasets/truck'
cols = 5
rows = 2

def load_data(directory):
    for item in os.listdir(directory):
        #print(item)
        if item.endswith('.jpg'):
            image = cv2.imread(directory+'/'+item)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            images.append(image)               
            labels.append(directory)  
    return images, labels     

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
                        

images1,labels1 = load_data(dir_car)   
images2,labels2 = load_data(dir_truck)

images = images1 + images2
images = np.array(images)

labels = labels1 + labels2

labelsTemp=[]
for label in labels:
    if label.endswith('car'):
        labelsTemp.append(0)
    else:
        labelsTemp.append(1)
labels = labelsTemp

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.25, random_state=42)


class_names = ['car', 'truck']

train_images = train_images / 255.0

test_images = test_images / 255.0

model=keras.Sequential([
    keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE,3)),
    keras.layers.Dense(128,activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions=model.predict(test_images)

num=rows*cols
plt.figure(figsize=(2*cols, 2*rows))
for i in range(num):
  plt.subplot(rows, cols, i+1)
  plot_image(i, predictions, test_labels, test_images) 
plt.show()

