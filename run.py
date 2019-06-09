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

def build_ladder_network(noise_std=0.3):
    # input_layer_l = Input((28, 28, 1))
    # input_layer_unlabelled = Input((28, 28, 1))
    input_layer_l = Input((784,))
    input_layer_unlabelled = Input((784,))

    Layer1 = Conv2D(32, 5, padding='valid', activation='relu')
    Layer12 = Conv2D(64, 5, padding='valid', activation='relu')
    Layer13 = Conv2D(64, 5, padding='valid', activation='relu')
    Layer14 = Conv2D(128, 5, padding='valid', activation='relu')

    Layer1t = Dense(128, use_bias=False , kernel_initializer='glorot_normal')
    Bias1t = BiasLayer()
    Layer2t = Dense(500, use_bias=False , kernel_initializer='glorot_normal')
    Bias2t = BiasLayer()
    Layer3t = Dense(250, use_bias=False , kernel_initializer='glorot_normal')
    Bias3t = BiasLayer()
    Layer4t = Dense(250, use_bias=False , kernel_initializer='glorot_normal')
    Bias4t = BiasLayer()
    Layer2 = Dense(250, use_bias=False , kernel_initializer='glorot_normal')
    Bias2 = BiasLayer()
    Layer3 = Dense(25, use_bias=False , kernel_initializer='glorot_normal')
    Bias3 = BiasLayer()

    def encode(input_layer, noise_std):
        input_layer_t = Reshape((28,28,1))(input_layer)
        h = add_noise(input_layer_t, noise_std)

        layer1_pre = Layer1(h)
        layer1_normalized = Lambda(batch_normalization)(layer1_pre)
        layer1_z = add_noise(layer1_normalized, noise_std)

        layer12_pre = Layer12(layer1_z)
        layer12_normalized = Lambda(batch_normalization)(layer12_pre)
        layer12_z = add_noise(layer12_normalized, noise_std)

        layer13_pre = Layer13(layer12_z)
        layer13_normalized = Lambda(batch_normalization)(layer13_pre)
        layer13_z = add_noise(layer13_normalized, noise_std)

        layer14_pre = Layer13(layer13_z)
        layer14_normalized = Lambda(batch_normalization)(layer14_pre)
        layer14_z = add_noise(layer14_normalized, noise_std)

        new_h = Flatten()(layer14_z)

        layer_1t_pre = Layer1t(new_h)
        layer_1t_normalized = Lambda(batch_normalization)(layer_1t_pre) 
        layer_1t_z = add_noise(layer_1t_normalized, noise_std)
        layer_1t = Activation('relu')(Bias1t(layer_1t_z))

        layer_2t_pre = Layer2t(layer_1t)
        layer_2t_normalized = Lambda(batch_normalization)(layer_2t_pre) 
        layer_2t_z = add_noise(layer_2t_normalized, noise_std)
        layer_2t = Activation('relu')(Bias2t(layer_2t_z))

        layer_3t_pre = Layer3t(layer_2t)
        layer_3t_normalized = Lambda(batch_normalization)(layer_3t_pre) 
        layer_3t_z = add_noise(layer_3t_normalized, noise_std)
        layer_3t = Activation('relu')(Bias3t(layer_3t_z))

        layer_4t_pre = Layer4t(layer_3t)
        layer_4t_normalized = Lambda(batch_normalization)(layer_4t_pre) 
        layer_4t_z = add_noise(layer_4t_normalized, noise_std)
        layer_4t = Activation('relu')(Bias4t(layer_4t_z))

        # layer_2_input = Reshape((7*7*64,), name='reshape1')(layer_13)
        layer_2_input = layer_4t
        layer_2_pre = Layer2(layer_2_input)
        layer_2_normalized = Lambda(batch_normalization)(layer_2_pre) 
        layer_2_z = add_noise(layer_2_normalized, noise_std)
        layer_2 = Activation('relu')(Bias2(layer_2_z))

        # layer_3_pre = Layer3(layer_2)
        layer_3_pre = Layer3(layer_1t)
        layer_3_normalized = Lambda(batch_normalization)(layer_3_pre) 
        layer_3_z = add_noise(layer_3_normalized, noise_std)
        layer_3 = Activation('softmax')(Bias3(layer_3_z))

        result = {
            'h': h,
            'layer1_z': layer1_z,
            'layer12_z': layer12_z,
            'layer13_z': layer13_z,
            'layer_1t_z': layer_1t_z,
            'layer_2t_z': layer_2t_z,
            'layer_3t_z': layer_3t_z,
            'layer_4t_z': layer_4t_z,
            'layer_2_z': layer_2_z,
            'layer_3_z': layer_3_z,
            'y': layer_3
        }

        return result

    result_l = encode(input_layer_l, 0.)
    result_l_c = encode(input_layer_l, 0.05)
    result_u = encode(input_layer_unlabelled, 0)
    result_u_c = encode(input_layer_unlabelled, 0.05)

    d_cost = []

    fc_dec = [
        Dense(784, use_bias=False , kernel_initializer='glorot_normal'),
        Conv2DTranspose(32, 5, padding='same'),
        Conv2DTranspose(64, 5, padding='same'),
        Conv2DTranspose(128, 5, padding='same'),
        Dense(784, use_bias=False , kernel_initializer='glorot_normal'),
        Dense(500, use_bias=False , kernel_initializer='glorot_normal'),
        Dense(250, use_bias=False , kernel_initializer='glorot_normal'),
        # Dense(250, use_bias=False , kernel_initializer='glorot_normal'),
        # Dense(250, use_bias=False , kernel_initializer='glorot_normal'),
    ]
    clean_z = [
        result_u['h'],
        result_u['layer1_z'],
        result_u['layer12_z'],
        result_u['layer13_z'],
        result_u['layer_1t_z'],
        result_u['layer_2t_z'],
        result_u['layer_3t_z'],
        # result_u['layer_4t_z'],
        # result_u['layer_2_z'],
        result_u['layer_3_z'],
    ]
    corr_z = [
        result_u_c['h'],
        result_u_c['layer1_z'],
        result_u_c['layer12_z'],
        result_u_c['layer13_z'],
        result_u_c['layer_1t_z'],
        result_u_c['layer_2t_z'],
        result_u_c['layer_3t_z'],
        # result_u_c['layer_4t_z'],
        # result_u_c['layer_2_z'],
        result_u_c['layer_3_z'],
    ]
    layer_sizes = [784, 32*3*3, 64*9, 64*9, 128 * 9, 128, 250, 250, 250, 25]
    denoising_cost = [1000, 10, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    for l in range(7, -1, -1):
        z, z_c = clean_z[l], corr_z[l]
        if l == 7:
            u = result_u_c['y']
        else:
            u = fc_dec[l]( z_est ) 
        u = Lambda(batch_normalization)(u)
        z_est  = DenoiseLayer()([z_c, u])
        layer_sizes[l] = 128
        denoising_cost[l] = 3 # TODO try reducing to 2 and 2.5.
        d_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_est  - z), 1)) / layer_sizes[l]) * denoising_cost[l])
        break
        if l == 4:
            z_est = Reshape((28,28,1))(z_est)
        if l == 0:
            z_est = Flatten()(z_est)

    # train_labels = Lambda(lambda x: x[0])([result_l_c['y'], result_l['y'], result_u_c['y'], result_u['y'], input_layer_u, input_layer_zest, result_u['h']])
    # train_labels = result_l_c['y']
    train_labels = Lambda(lambda x: x[0])([result_l_c['y'], result_l['y'], result_u_c['y'], result_u['y'], u, z_est, z])

    model = Model([input_layer_l, input_layer_unlabelled], train_labels)
    model.add_loss(tf.add_n(d_cost))

    test_model = Model(input_layer_l, result_l['y'])
    model.test_model = test_model

    return model


x_train, y_train = parse_csv('sign_mnist_train.csv')
x_test, y_test = parse_csv('sign_mnist_test.csv')

x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# only select limited number of labels 
sampled = range(x_train.shape[0])
random.seed(0)
random.shuffle(sampled)
sampled = sampled[:250]

# x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
# x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

x_train_unlabeled = x_train
x_train_labeled = x_train[sampled]
y_train_labeled = y_train[sampled]


n_rep = x_train_unlabeled.shape[0] / x_train_labeled.shape[0]
x_train_labeled_rep = np.concatenate([x_train_labeled] * n_rep)[:10000]
y_train_labeled_rep = np.concatenate([y_train_labeled] * n_rep)[:10000]
x_train_unlabeled = x_train_unlabeled[:10000]


model = build_ladder_network()

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.compile(tf.keras.optimizers.Adam(lr=0.001) , 'categorical_crossentropy', metrics=['accuracy'])
print model.summary()
for _ in range(100):
    model.fit([x_train_labeled_rep, x_train_unlabeled], y_train_labeled_rep, epochs=1, batch_size=32)
    y_test_pr = model.test_model.predict(x_test , batch_size=100 )
    model.test_model.save('model_r.h5')
    foomodel = load_model('model_r.h5', custom_objects={'BiasLayer':BiasLayer, 'DenoiseLayer':DenoiseLayer})
    print "test accuracy" , accuracy_score(y_test.argmax(-1) , y_test_pr.argmax(-1))