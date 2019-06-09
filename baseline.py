from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import keras
from sklearn.metrics import accuracy_score

import tensorflow as tf

import csv

import random
import numpy as np

from time import time
from keras.callbacks import TensorBoard

def parse_csv(filename):
    f = open(filename)
    linesr = f.readlines()
    f.close()
    total = 0
    first = True
    lines = []
    labels = []
    for line in linesr:
        if first:
            first = False
            continue
        line = line.strip().split(',')
        line = np.array(map(lambda x: int(x), line))
        labels.append(line[0])
        lines.append(line[1:])
        total += 1
        if total >= 20000:
            break
    return np.array(lines), np.array(labels)


x_train, y_train = parse_csv('sign_mnist_train.csv')
x_test, y_test = parse_csv('sign_mnist_test.csv')

x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

sampled = range(x_train.shape[0])
random.seed(0)
random.shuffle(sampled)
sampled = sampled[:1000]

# x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
# x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

x_train = x_train[sampled]
y_train = y_train[sampled]

model = Sequential()
model.add(Reshape((28,28,1), input_shape=(784,)))
model.add(Conv2D(32, kernel_size=3, padding='valid', activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(64, kernel_size=3, padding='valid', activation='relu'))
model.add(Conv2D(64, kernel_size=3, padding='valid', activation='relu'))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(25, activation='softmax'))

model.compile(tf.keras.optimizers.Adam(lr=0.001) , 'categorical_crossentropy', metrics=['accuracy'])
print model.summary()
for _ in range(100):
    model.fit(x_train, y_train, batch_size=128, epochs=20)
    model.save('model.h5')
    y_test_pr = model.predict(x_test, batch_size=100)
    print "test accuracy" , accuracy_score(y_test.argmax(-1) , y_test_pr.argmax(-1))