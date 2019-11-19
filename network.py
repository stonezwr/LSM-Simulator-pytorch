import reservoir
import feedforward
from tqdm import tqdm
from scipy.sparse import coo_matrix
import numpy as np


def lsm(n_inputs, n_classes, n_steps, epoches, x_train, x_test, y_train, y_test):
    stdp = False  # incomplete
    dim1 = [3, 3, 15]
    r1 = reservoir.ReservoirLayer(n_inputs, 135, n_steps, dim1, is_input=True)
    s1 = feedforward.SpikingLayer(135, n_classes, n_steps)
    accuracy = 0
    best_acc_e = 0
    
    # train stdp
    if stdp:
        r1.stdp = True
        print("start stdp for reservoir")
        for i in range(20):
            for i in tqdm(range(len(x_train))):
                x = np.asarray(x_train[i].todense())
                r1.forward(x)
        r1.stdp = False
        print("finish stdp")

    for e in range(epoches):
        for i in tqdm(range(len(x_train))):  # train phase
            x = np.asarray(x_train[i].todense())
            o_r1 = r1.forward(x)
            o_s1 = s1.forward(o_r1, e, y_train[i])
            fire_count = np.sum(o_s1, axis=0)
            # print(y_train[i])
            # print(fire_count)

        correct = 0
        for i in tqdm(range(len(x_test))):  # test phase
            x = np.asarray(x_test[i].todense())
            o_r1 = r1.forward(x)
            o_s1 = s1.forward(o_r1)

            fire_count = np.sum(o_s1, axis=0)
            # print(fire_count)
            index = np.argmax(fire_count)
            if index == y_test[i]:
                correct = correct + 1
        acc = correct / len(x_test)
        print("test accuracy at epoch %d is %0.2f%%" % (e, acc * 100))
        if accuracy < acc:
            accuracy = acc
            best_acc_e = e
    return accuracy, best_acc_e
