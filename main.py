import os
import sys
import loadDataset
import network
import torch
import random


def shuffle_dataset(data, label):
    c = list(zip(data, label))
    random.shuffle(c)
    data, label = zip(*c)
    return data, label


if __name__ == "__main__":
    n_steps = 700
    n_channels = 78
    n_classes = 26
    data_set = "TI46"

    classifier = "svmcv"

    dtype = torch.float

    # Check whether a GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if data_set == "MNIST":
        data_path = os.path.expanduser("./mnist")

        trainset, testset, classes = loadDataset.get_mnist(data_path)

        # Standardize data
        x_train = torch.tensor(trainset.train_data, device=device, dtype=dtype)/255

        x_test = torch.tensor(testset.test_data, device=device, dtype=dtype)/255

        y_train = torch.tensor(trainset.train_labels, device=device, dtype=dtype)
        y_test = torch.tensor(testset.test_labels, device=device, dtype=dtype)

    elif data_set == "TI46":
        speaker_per_class = 1
        data_path = os.path.expanduser("./TI46_alpha")
        x_train, x_test, y_train, y_test = loadDataset.loadTI46Alpha(device, data_path, speaker_per_class, n_steps,
                                                                     n_channels, dtype)
    else:
        print("please choose correct dataset, MNIST or TI46")
        sys.exit(-1)

    if classifier == "calcium_supervised":
        x_train, y_train = shuffle_dataset(x_train, y_train)

    accuracy, e = network.lsm(device, n_channels, n_classes, n_steps, x_train, x_test, y_train, y_test, classifier,
                              dtype)
    print("best accuracy: %0.2f%% is achieved at epoch %d" % (accuracy*100, e))



