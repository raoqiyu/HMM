# -*- coding: utf-8 -*-
# Author     : raoqiyu@gmail.com
# Time      : 2019-05-05 10:37
# FileName  : hmm.py

import numpy as np


class HMM:
    def __init__(self, M, V):
        """

        Parameters:
        m - the number of hidden states

        """

        self.M = M
        self.V = V

        self.pi = np.ones(self.M) / self.M # initial state distribution

        self.A = np.random.random((self.M, self.M)) # state transition matrix
        self.A /= self.A.sum(axis=1,keepdims=True)

        self.B = np.random.random((self.M,self.V)) # observed state matrix
        self.B /= self.B.sum(axis=1, keepdims=True)

    def forward(self, x):
        """
        the forward part of the forward-backward algorithm
        calculate forward probability: the probability of observed sequence at time t is  (x1,x2,x3...,xt) and the state
            at time t is qi given model λ
        α_t(i) = P(x1,x2,...xt,i_t=q_i | λ)


        Parameters:
        x - observed sequence, np.array, T*1
        Returns:
        alpha - forward probability
        """
        T = x.shape[0]

        alpha = np.zeros((x.shape[0],self.M))
        alpha[0] = self.pi*self.B[:,x[0]]
        for t in range(1,T):
            alpha[t] = alpha[t-1].dot(self.A) * self.B[:,x[t]]

        return alpha

    def backward(self, x):
        """
        the backward part of the forward-backward algorithm
        calculate backward probability: the probability of the observed sequence from time t+1 to T is
            (x_t+1,x_t+2,...,x_T) given state is qi at time t and the model λ
        α_t(i) = P(x_t+1,x_t+2,...x_T|λ, i_t=q_i )

        Parameters:
        x - observed sequence
        Returns:
        beta - backward probability
        """
        T = x.shape[0]

        beta = np.zeros((T,self.M))
        beta[-1] = 1
        for t in range(T-2,-1,-1):
            beta[t] = self.A.dot(self.B[:,x[t]]) * beta[t+1]

        return beta

    def calc_gamma_per_element(self, t, i, alpha, beta):
        """
        calculate probability of state qi at time t given model λ and  observed  sequence x
        γ_t(i) = p(i_t = q_i | x, λ)

        Note : P(X|λ) = sum(P(α_t(i)), i=1,2,...V
        Parameters:
        x - observed sequence
        alpha - forward probability
        beta - backward probability
        Returns:
        γ
        """
        gamma_numerator = alpha[t,i]*beta[t,i]
        gamma_denominator = alpha[-1].sum()

        return gamma_numerator/gamma_denominator

    def calc_gamma(self, t, i, alpha, beta):
        """
        calculate probability of state qi at time t given model λ and  observed  sequence x
        γ_t(i) = p(i_t = q_i | x, λ)

        Note : P(X|λ) = sum(P(α_t(i)), i=1,2,...V
        Parameters:
        x - observed sequence
        alpha - forward probability
        beta - backward probability
        Returns:
        γ
        """

        gamma_numerator = alpha * beta
        gamma_denominator = alpha[-1].sum()

        return gamma_numerator/gamma_denominator

    def calc_ksi(self, x, alpha, beta):
        """
        calculate probability of state qi at time t and state qj at time t+1 given model and  observed  sequence x
        ξ_t(i,j) = p(i_t = q_i, t_t+1 = q_j  | x, λ)
        Parameters:
        x - observed sequence
        alpha - forward probability
        beta - backward probability
        Returns:
        ξ
        """
        T = alpha.shape[0]

        ksi_numerator = np.zeros((T,self.M,self.M))
        ksi_denominator = alpha[-1].sum()
        for t in range(T):
            for i in range(self.M):
                for j in range(self.M):
                    ksi_numerator[t,i,j] = alpha[t,i] * self.A[i,j] * self.B[j,x[t+1]] * beta[t+1,j]

        return ksi_numerator/ksi_denominator

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

    def fit(self, X, max_iter=10):
        """
        train  HMM  using the Baum-Welch algorithm
        a specific instance of the expectation-maximization algorithm

        Parameters:
        X - array of observed sequence
        max_iter - max iteration of training
        """

