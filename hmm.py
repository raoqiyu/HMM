# -*- coding: utf-8 -*-
# Author     : raoqiyu@gmail.com
# Time      : 2019-05-05 10:37
# FileName  : hmm.py

import time
import numpy as np


class HMM:
    def __init__(self, M, V):
        """
        Hidden Markov Model for Discrete Data
        Parameters:
        m - the number of hidden states
        v - the number of observed states

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
            beta[t] = self.A.dot(self.B[:,x[t+1]] * beta[t+1])

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
        γ_t(i)
        """
        gamma_numerator = alpha[t,i]*beta[t,i]
        gamma_denominator = alpha[-1].sum()

        return gamma_numerator/gamma_denominator

    def calc_gamma(self, alpha, beta):
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
        for t in range(T-1):
            for i in range(self.M):
                for j in range(self.M):
                    ksi_numerator[t,i,j] = alpha[t,i] * self.A[i,j] * self.B[j,x[t+1]] * beta[t+1,j]

        return ksi_numerator/ksi_denominator

    def fit(self, X, max_iter=10):
        """
        train  HMM  using the Baum-Welch algorithm
        a specific instance of the expectation-maximization algorithm

        Parameters:
        X - array of observed sequence
        max_iter - max iteration of training
        """
        begin_time = time.time()

        n_samples,T = X.shape[0],X.shape[1]

        costs = []
        P = np.zeros((n_samples,1)) # probability of observed sequence P(X[i]|λ)

        for i_iter in range(max_iter):

            # Step 1
            # forward and backward
            alphas, betas = [], []
            for i_sample in range(n_samples):
                alpha = self.forward(X[i_sample]) # T * M
                beta = self.backward(X[i_sample]) # T * M
                alphas.append(alpha)
                betas.append(beta)
                P[i_sample] = alpha[-1].sum()

            # record costs
            costs.append(np.sum(np.log(P)))

            # Step 2
            # re-estimate pi, A, B

            # Step 2.1 re-estimate pi (mean value)
            self.pi =  np.sum((alphas[i_sample][0] * betas[i_sample][0])/P[i_sample]  \
                                for i_sample in range(n_samples)) / n_samples

            # Step 2.2 re-estimate A, B
            tmp_A, tmp_B = [],[]
            for i_sample in range(n_samples):
                # Step 2.2.1  update A
                a_ksi = self.calc_ksi(X[i_sample], alphas[i_sample],betas[i_sample]) # T-1 * M * M
                a_gamma  = self.calc_gamma(alphas[i_sample][:-1],betas[i_sample][:-1]) # T-1 * M

                A_numerator = np.sum(a_ksi, axis=0) # M*M
                A_denominator = np.sum(a_gamma, axis=0) # M*1
                tmp_A.append(A_numerator/A_denominator)

                # Step 2.2.1  update B
                B_numerator = np.zeros((self.M,self.V)) # M * V
                for j in range(self.M):
                    for k in range(self.V):
                        B_gamma_numerator = 0
                        for t in range(T):
                            if X[i_sample][t] == k:
                                B_gamma_numerator += self.calc_gamma_per_element(t,j,
                                                                                 alphas[i_sample],betas[i_sample])
                        B_numerator[j,k] = B_gamma_numerator

                b_gamma = self.calc_gamma(alphas[i_sample],betas[i_sample]) # T * M
                B_denominator = np.sum(b_gamma, axis=0)  # M*1
                tmp_B.append(B_numerator / B_denominator) # M*V

            self.A = np.mean(tmp_A,axis=0) # M*M
            self.B = np.mean(tmp_B, axis=0)  # M*V

        end_time = time.time()
        print('Fit duration: ', end_time - begin_time)

    def viterbi(self,x):
        """
        Viterbi algorithm
        Calculate the most likely hidden state sequence given observed sequence x
        δ : delta
        ψ : psai

        Parameter:
        x -  observed sequence
        Return:
        states - most likely hidden state sequence
        """
        T = x.shape[0]

        # Step 1: initialize delta and psai
        delta = np.zeros((T,self.M))
        delta[0] = self.pi * self.B[:, x[0]]

        psai  = np.zeros((T,self.M))

        # Step 2: update delta and psai
        for t in range(1, T):
            for j in range(self.M):
                delta[t,j] = np.max(delta[t-1]*self.A[:,j]) * self.B[j, x[t]]
                psai[t,j] = np.argmax(delta[t-1]*self.A[:,j])

        # Step 3: backtrack
        states = np.zeros(T)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psai[t+1,states[t+1]]

        return states
