import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import csv

import random
import numpy as np

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


x_test, y_test = parse_csv('sign_mnist_test.csv')

for i in range(1000, len(y_test)):
    if y_test[i] == 17:
        plt.imshow(x_test[i].reshape((28, 28)), cmap="gray")
        plt.savefig('R.png')
        exit()