import os
import sys
import torchvision.datasets
import torchvision.transforms as transforms
import numpy as np
from scipy.sparse import coo_matrix


def get_mnist(data_path):
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    trainset = torchvision.datasets.MNIST(data_path, train=True, transform=transform, download=True)
    testset = torchvision.datasets.MNIST(data_path, train=False, transform=transform, download=True)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    return trainset, testset, classes


def loadTI46Alpha(data_path, num_per_class, n_steps, n_channels, dtype):
    if not os.path.exists(data_path):
        print('Given path {} not found'.format(data_path))
        sys.exit(-1)
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(26):
        pathname = os.path.join(data_path, str(i))
        count = 0
        for fn in os.listdir(pathname):
            if fn[0] == '.':
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
                if count < (0.8 * num_per_class):
                    x_train.append(spike)
                    y_train.append(label)
                else:
                    x_test.append(spike)
                    y_test.append(label)
            count = count + 1
    return x_train, x_test, y_train, y_test


