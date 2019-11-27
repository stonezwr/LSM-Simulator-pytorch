import os
import sys
import torchvision.datasets
import torchvision.transforms as transforms
import numpy as np
from scipy.sparse import coo_matrix
import random


def get_mnist(data_path):
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    trainset = torchvision.datasets.MNIST(data_path, train=True, transform=transform, download=True)
    testset = torchvision.datasets.MNIST(data_path, train=False, transform=transform, download=True)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    return trainset, testset, classes


def genrate_poisson_spikes(data, n_steps):
    row = []
    col = []
    val = []
    n_channels = len(data)
    for i in range(n_channels):
        for j in range(n_steps):
            if random.random() < data[i]:
                row.append(j)
                col.append(i)
                val.append(1)

    spike = coo_matrix((val, (row, col)), shape=(n_steps, n_channels))
    return spike


def loadTI46Alpha(data_path, speaker_per_class, n_steps, n_channels):
    if not os.path.exists(data_path):
        print('Given path {} not found'.format(data_path))
        sys.exit(-1)
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(26):
        pathname = os.path.join(data_path, str(i))
        for fn in os.listdir(pathname):
            if fn[0] == '.':
                continue
            if speaker_per_class <= 8 and (fn[0] == 'm' or int(fn[1]) > speaker_per_class):
                continue
            elif 8 < speaker_per_class <= int(fn[1])+8 and fn[0] == 'm':
                continue

            row = []
            col = []
            val = []
            filename = os.path.join(pathname, fn)
            if os.path.isfile(filename):
                label = i
                with open(filename, 'r') as f:
                    data = np.loadtxt(f, dtype=int)
                    nrows, ncols = data.shape
                    for j in range(nrows):
                        if data[j, 0] == -1 or data[j, 1] >= n_steps:
                            continue
                        row.append(data[j, 1])
                        col.append(data[j, 0])
                        val.append(1)

                spike = coo_matrix((val, (row, col)), shape=(n_steps, n_channels))
                if int(fn[4]) == 3 or int(fn[4]) == 2:
                    x_test.append(spike)
                    y_test.append(label)
                else:
                    x_train.append(spike)
                    y_train.append(label)
    return x_train, x_test, y_train, y_test
