from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import datasets, layers, models, optimizers
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

X = X.reshape((1372, 64, 64, 1))
print('X shape: ', X.shape, 'y shape: ', y.shape)


# # Data division

from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)
#train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=1)

train_data.shape


# # Convolutional base

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.summary()


# # Dense layer at the top

model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(2, activation='sigmoid'))

# # Train model
rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.99)
adagrad = optimizers.Adagrad(lr=0.01, epsilon=1e-6, decay=0.99)
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.9, nesterov=True)

model.compile(optimizer=adam,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data, train_labels, epochs=20)#, validation_data=(val_data,val_labels))


# # Test model

test_loss, test_acc = model.evaluate(test_data, test_labels)

# Save test accuracy
difference = abs(history.history["acc"][-1] - test_acc)
result = "Test accuracy = " + str(test_acc) + ", Difference between test_acc and train_acc: " + str(difference) + ".\n"
print(result)
with open('history.txt', 'a') as myfile:
    myfile.write(result)

# Evaluate
import matplotlib.pyplot as plt

f = plt.figure(figsize=(10,6))
plt.plot(history.history["loss"], label="train loss")
#plt.plot(history.history["val_loss"], label="val_loss")
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

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

