from tqdm import tqdm
import sys
import torch

import feedforward
import reservoir
import svm


def lsm(device, n_inputs, n_classes, n_steps, x_train, x_test, y_train, y_test, classifier, dtype):
    stdp = True  # stdp enabled
    dim1 = [3, 3, 15]
    r1 = reservoir.ReservoirLayer(device, n_inputs, 135, n_steps, dim1, is_input=True, homeostasis=True, dtype=dtype)
    s1 = feedforward.SpikingLayer(device, 135, 100, n_steps, dtype=dtype)

    # train stdp
    if stdp:
        r1.stdp_i = True
        r1.stdp_r = True
        s1.stdp_i = True
        print("start stdp")
        for e_stdp in range(10):
            for i in tqdm(range(len(x_train))):
                r1.reset()
                x = x_train[i].to_dense()
                o_r1 = r1.forward(x)
                s1.forward(o_r1)
        print("finish stdp")
        r1.stdp_i = False
        r1.stdp_r = False
        s1.stdp_i = False

    if classifier == "svmcv":
        samples = []
        label = []
        for i in tqdm(range(len(x_train))):  # train phase
            r1.reset()
            x = x_train[i].to_dense()
            o_r1 = r1.forward(x)
            fire_count = torch.sum(o_r1, dim=0)
            samples.append(fire_count.numpy())
            label.append(y_train[i])

        for i in tqdm(range(len(x_test))):  # test phase
            r1.reset()
            x = x_test[i].to_dense()
            o_r1 = r1.forward(x)
            fire_count = torch.sum(o_r1, dim=0)
            samples.append(fire_count.numpy())
            label.append(y_test[i])

        accuracy = svm.cvSVM(samples, label, 5)
        best_acc_e = 0
    else:
        print('Given classifier {} not found'.format(classifier))
        sys.exit(-1)
    return accuracy, best_acc_e
