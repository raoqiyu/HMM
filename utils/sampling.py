# -*- coding: utf-8 -*-
# Author     : raoqiyu@gmail.com
# Time      : 2019-05-06 17:51
# FileName  : sampling.py
import numpy as np


def reject_sampling(probs, k=3):
    """
    reject sampling

    :param probs: array[], probability for every item
    :param k:
    :return: the item sampled
    """
    stop = False
    i_sample = -1
    while not stop:
        z = np.random.uniform(0, 1)
        u = np.random.uniform(0, k * z)
        i_sample = np.random.choice(probs.shape[0], p=probs)
        pz = probs[i_sample]
        if pz >= u:
            stop = True
    return i_sample


def random_sampling(probs):
    """
    Think of the probs as a bunch of buckets, and then randomly generate a number to see which bucket it will fall into.

    :param probs: array[], probability for every item
    :return: the item sampled
    """
    rand_probability = np.random.uniform(0,1)
    cum_prob = 0
    i_sample = -1
    for i, prob in enumerate(probs):
        cum_prob += prob
        if rand_probability < cum_prob:
            i_sample = i
            break
    return i_sample
