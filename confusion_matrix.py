import matplotlib
matplotlib.use('Agg')
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

from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

class BiasLayer( Layer):
    def __init__(self  , **kwargs):
        super(BiasLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        if self.built:
            return
        self.bias = self.add_weight(name='bias', shape= input_shape[1:], initializer='zeros', trainable=True)
        self.built = True

        super(BiasLayer, self).build(input_shape)  
        
    def call(self, wx, training=None):
        return tf.add(wx,  self.bias)


class DenoiseLayer(Layer):
    def __init__(self , **kwargs):
        super(DenoiseLayer, self).__init__(**kwargs)
        
    def weight(self, init, name):
        if init == 0:
            return self.add_weight(name='denoise_'+name, shape=( self.size,), initializer='zeros', trainable=True)
        elif init == 1:
            return self.add_weight(name='denoise_'+name, shape=(self.size,), initializer='ones', trainable=True)

    def build(self, shape):
        self.size = shape[0][-1]
        
        self.a1 = self.weight(0., 'a1')
        self.a2 = self.weight(1., 'a2')
        self.a3 = self.weight(0., 'a3')
        self.a4 = self.weight(0., 'a4')
        self.a5 = self.weight(0., 'a5')

        self.a6 = self.weight(0., 'a6')
        self.a7 = self.weight(1., 'a7')
        self.a8 = self.weight(0., 'a8')
        self.a9 = self.weight(0., 'a9')
        self.a10 = self.weight(0., 'a10')
        
        super(DenoiseLayer, self).build(shape)

    def call(self, x):
        z_c, u = x 
        
        a1 = self.a1 
        a2 = self.a2 
        a3 = self.a3 
        a4 = self.a4 
        a5 = self.a5 
        a6 = self.a6 
        a7 = self.a7 
        a8 = self.a8 
        a9 = self.a9 
        a10 =self.a10
        
        mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
        v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

        z_est = (z_c - mu) * v + mu
        return z_est
    
    def compute_output_shape(self, shape):
        return (shape[0][0], self.size)



def batch_normalization(batch, mean=None, var=None):
    if mean is None or var is None:
        mean, var = tf.nn.moments(batch, axes=[0])
    return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

def add_noise(inputs, noise_std):
    return Lambda(lambda x: x + tf.random_normal(tf.shape(x)) * noise_std)(inputs)


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
# y_test = keras.utils.to_categorical(y_test)

model = load_model('model_r.h5', custom_objects={'BiasLayer':BiasLayer, 'DenoiseLayer':DenoiseLayer})

from sklearn.metrics import confusion_matrix
import numpy as np

labels = y_test
predictions = model.predict(x_test)

cm = confusion_matrix(labels, predictions.argmax(axis=1))
print cm
recall = np.diag(cm) * 1.0 / np.sum(cm, axis = 1)
precision = np.diag(cm) * 1.0 / np.sum(cm, axis = 0)

print np.mean(recall)
print np.mean(precision)

classes = list("ABCDEFGHIKLMNOPQRSTUVWXY")
print classes
fig, ax = plt.subplots()
cmap = plt.cm.Blues
im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
ax.figure.colorbar(im, ax=ax)

ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=classes, yticklabels=classes,
       title='Sign Language Classification Confusion Matrix',
       ylabel='True label',
       xlabel='Predicted label')

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
# normalize = True
# fmt = '.2f' if normalize else 'd'
# thresh = cm.max() / 2.
# for i in range(cm.shape[0]):
#     for j in range(cm.shape[1]):
#         ax.text(j, i, format(cm[i, j], fmt),
#                 ha="center", va="center",
#                 color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()

plt.savefig('plot.png')