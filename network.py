import reservoir
import feedforward
from tqdm import tqdm
from scipy.sparse import coo_matrix
import numpy as np


def lsm(n_inputs, n_classes, n_steps, epoches, x_train, x_test, y_train, y_test):
    dim1 = [3, 3, 15]
    r1 = reservoir.ReservoirLayer(n_inputs, 135, n_steps, dim1, is_input=True)
    s1 = feedforward.SpikingLayer(135, n_classes, n_steps)
    accuracy = 0
    best_acc_e = 0

    for e in range(1):
        for i in tqdm(range(len(x_train))):  # train phase
            x = np.asarray(x_train[i].todense())
            o_r1 = r1.forward(x)
            o_s1 = s1.forward(o_r1, e, y_train[i])
            fire_count = np.sum(o_s1, axis=0)
            print(y_train[i])
            print(fire_count)

        correct = 0
        for i in tqdm(range(len(x_test))):  # test phase
            x = np.asarray(x_test[i].todense())
            o_r1 = r1.forward(x)
            o_s1 = s1.forward(o_r1)

            fire_count = np.sum(o_s1, axis=0)
            print(fire_count)
            index = np.argmax(fire_count)
            if index == y_test[i]:
                correct = correct + 1
        acc = correct / len(x_test)
        print("test accuracy at epoch %d is %0.2f%%" % (e, acc * 100))
        if accuracy < acc:
            accuracy = acc
            best_acc_e = e
    return accuracy, best_acc_e
