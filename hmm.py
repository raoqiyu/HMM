# -*- coding: utf-8 -*-
# Author     : raoqiyu@gmail.com
# Time      : 2019-05-05 10:37
# FileName  : hmm.py

import numpy as np

np.array

class HMM:
    def __init__(self):
        pass

    def forward(self, x):
        """
        the forward part of the forward-backward algorithm
        calculate forward probability

        Parameters:
        x -  observed sequence
        Returns:
        alpha -   forward probability
        """

        pass

    def backward(self, x):
        """
        the backward part of the forward-backward algorithm
        calculate backward probability

        Parameters:
        x -  observed sequence
        Returns:
        beta -   backward probability
        """

        pass

    def viterbi(self,x):
        """
        Viterbi algorithm
        returns the most likely state sequence given observed sequence x

        Parameter:
        x -  observed sequence
        Return:
        states - state sequence
        """
        pass

    def calc_gamma(self):
        """
        calculate probability of state qi at time t given model λ and  observed  sequence x
        γ_t(i) = p(i_t = q_i | x, λ)
        Parameters:

        Returns:
        """
        pass

    def calc_ksi(self):
        """
        calculate probability of state qi at time t and state qj at time t+1 given model and  observed  sequence x
        ξ_t(i,j) = p(i_t = q_i, t_t+1 = q_j  | x, λ)
        Parameters:

        Returns:
        """

    def fit(self, X, max_iter=10):
        """
        train  HMM  using the Baum-Welch algorithm
        a specific instance of the expectation-maximization algorithm

        Parameters:
        X - array of observed sequence
        max_iter - max iteration of training
        """
        pass
