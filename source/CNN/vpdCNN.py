from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
print(tf.__version__)


# # Loading data

import json
import numpy as np
import os
from random import shuffle

datadir = 'source/CNN/data/'
fulldir = os.path.join(os.getcwd(), datadir)
allfiles = os.listdir(fulldir)

dataset = []

for filename in allfiles:
    filepath = os.path.join(datadir, filename)
    name = 'source/CNN/data/' + filename.split(".")[0]

    with open(name + '.json', 'r') as file:
        person = json.load(file)
    dataset.append(person)
    
X = []
y = []

for person in dataset:
    spectogram = person['spectogram']/np.float32(255) #normalize input pixels 
    status = int(person['status'])
    X.append(spectogram)
    y.append(status)
X = np.array(X)
y = np.array(y)

X = X.reshape((1372, 28, 28, 1))
print('X shape: ', X.shape, 'y shape: ', y.shape)


# # Data division

from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=1)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=1)

train_data.shape


# # Convolutional base

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.summary()


# # Dense layer at the top

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='sigmoid'))

model.summary()


# # Train model

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data, train_labels, epochs=10, validation_data=(val_data,val_labels))


# # Test model

test_loss, test_acc = model.evaluate(test_data, test_labels)
print(test_acc)

# Evaluate
import matplotlib.pyplot as plt

f = plt.figure(figsize=(10,6))
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel('epoch', fontsize=14)
plt.ylabel('loss', fontsize=14)
plt.title('loss history', fontsize=14)
plt.legend(fontsize='large')

f.savefig('loss_history.pdf', bbox_inches='tight')

f = plt.figure(figsize=(10,6))
plt.plot(history.history["acc"])
plt.xlabel('epoch', fontsize=14)
plt.ylabel('acc', fontsize=14)

f.savefig('accuracy.pdf', bbox_inches='tight')
