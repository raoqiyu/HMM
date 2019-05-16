# -*- coding: utf-8 -*-
# Author     : raoqiyu@gmail.com
# Time      : 2019-05-05 17:41
# FileName  : test_HMM.py
import math
import numpy as np
import matplotlib.pyplot as plt
from hmm import HMM


def fit_coin():
    """
    example from https://github.com/lazyprogrammer/machine_learning_examples/blob/master/hmm_class/hmmd.py
    :return:
    """
    X = []
    for line in open('./data/coin_data.txt'):
        # 1 for H, 0 for T
        x = [1 if e == 'H' else 0 for e in line.rstrip()]
        X.append(x)
    X = np.array(X)
    print(X.shape)

    hmm = HMM(2,2)
    hmm.fit(X,max_iter=50)
    L = hmm.log_likelihood(X).sum()
    print("LL with fitted params:", L)

    # try true values
    hmm.pi = np.array([0.5, 0.5])
    hmm.A = np.array([[0.1, 0.9], [0.8, 0.2]])
    hmm.B = np.array([[0.6, 0.4], [0.3, 0.7]])
    L = hmm.log_likelihood(X).sum()
    print("LL with true params:", L)

    # try viterbi
    print("Best state sequence for:", X[0])
    print(hmm.viterbi(X[0]))


def show_data(x,y):
    plt.plot(x, y, 'g')
    plt.show()


def sin(length):
    """
    example from https://github.com/WenDesi/lihang_book_algorithm/blob/master/hmm/hmm.py
    :param length:
    :return:
    """
    X = [i for i in range(length)]
    Y = [int(math.sin((x%20/10*math.pi))*50)+50 for x in X]
    return X,Y


def fit_sin(length=100,n_hidden_states=10,max_iter=10):
    x, y = sin(length)

    hmm = HMM(n_hidden_states,101)
    hmm.fit(np.reshape(y,(1,-1)), max_iter=max_iter)
    y_gen = hmm.generate(length)

    show_data(x,y)
    show_data(x,y_gen)


if __name__ == '__main__':
    # fit_coin()

    # 100, 20, 100 : good
    # 100, 30, 100 : bad
    # 100, 40, 100 : good
    fit_sin(100,40,100)


