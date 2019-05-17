# -*- coding: utf-8 -*-
# Author     : raoqiyu@gmail.com
# Time      : 2019-05-05 10:37
# FileName  : hmm.py

import time
import numpy as np
from functools import reduce
np.random.seed(1024)

from utils.sampling import reject_sampling, random_sampling

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

        scaled version:
            α_1(i) = P(x1,i_1=q_i | λ)
        scaled α_1(i) = P(x1,i_1=q_i | λ)/sum(P(x1,i_1=q_i | λ), i=1,...M)
        scaled factor c_1 = 1//sum(P(x1,i_1=q_i | λ)

        scaled α_1(i) = α_1(i) * (scaled factor c_1)

        scaled α_t(i) = sum( scaled_α_t-1(j) * A[j,i] * B[i,xt], j=1,..,M) * (scaled factor c_t)
            scaled factor c_t = sum(sum( scaled_α_t-1(j) * A[j,i] * B[i,xt], j=1,..,M), i=1,..,M)

        so
            scaled α_1(i) = α_1(i) * (scaled factor c_1)
            scaled α_2(i) = sum( scaled_α_1(j) * A[j,i] * B[i,xt], j=1,..,M) * (scaled factor c_2)
                          = sum( α_1(i) * (scaled factor c_1) * A[j,i] * B[i,xt], j=1,..,M) * (scaled factor c_2)
                          = sum( α_1(i) * A[j,i] * B[i,xt], j=1,..,M) * (scaled factor c_2) * (scaled factor c_1)
                          = α_2(i) * (scaled factor c_2) * (scaled factor c_1)
            scaled α_3(i) = sum( scaled_α_2(j) * A[j,i] * B[i,xt], j=1,..,M) * (scaled factor c_3)
                          = sum(α_2(i) * (scaled factor c_2) * (scaled factor c_1) * A[j,i] * B[i,xt], j=1,..,M) * (scaled factor c_3)
                          = sum(α_2(i) * A[j,i] * B[i,xt], j=1,..,M) * (scaled factor c_3) * (scaled factor c_2) * (scaled factor c_1)
                          =  α_3(i) * (scaled factor c_3) * (scaled factor c_2) * (scaled factor c_1)

            scaled α_t(i) = prod(c_j, j=1,..,t) * α_t(i)

        1 = sum(scaled_α_T(i),i =1,...,M) = sum( prod(ct, t=1,..,T) * α_T(i)) = prod(ct, t=1,..,T) * sum(α_T(i),i=1,..,M)
          = prod(ct, t=1,..,T) * P(x|λ)

        P(x|λ) = 1 / prod(ct, t=1,..,T)

        α_t(i)
        Parameters:
        x - observed sequence, np.array, T*1
        Returns:
        alpha - forward probability
        """
        T = x.shape[0]

        alpha = np.zeros((x.shape[0],self.M))
        scale = np.zeros((x.shape[0],1))
        alpha[0] = self.pi*self.B[:,x[0]]
        scale[0] = 1/alpha[0].sum()
        alpha[0] *= scale[0]
        for t in range(1,T):
            alpha[t] = alpha[t-1].dot(self.A) * self.B[:,x[t]]
            scale[t] = 1/alpha[t].sum()
            alpha[t] *= scale[t]

        return alpha, scale

    def backward(self, x, scale=None):
        """
        the backward part of the forward-backward algorithm
        calculate backward probability: the probability of the observed sequence from time t+1 to T is
            (x_t+1,x_t+2,...,x_T) given state is qi at time t and the model λ
        β_t(i) = P(x_t+1,x_t+2,...x_T|λ, i_t=q_i )


        scaled version:
        scaled βT(i) = cT * βT(i)
        scaled βt(i) = sum( A[i,j] * B[j, xt+1] * scaled_βt+1(j), j=1,..,M)


        so,
            scaled βT(i) = cT * βT(i)
            scaled βT-1(i) = cT-1 * sum( A[i,j] * B[j, xt+1] * scaled_βT(j), j=1,..,M)
                           = sum( A[i,j] * B[j, xt+1] * cT * βT(i), j=1,..,M)
                           = cT-1 * cT * sum( A[i,j] * B[j, xt+1] * βT(i), j=1,..,M)
                           = cT-1 * cT * βT-1(i)
            scaled βT-2(i) = cT-2 * sum( A[i,j] * B[j, xt+1] * scaled_βT-1(j), j=1,..,M)
                           = cT-2 * sum( A[i,j] * B[j, xt+1] * cT-1 * cT * βT-1(i), j=1,..,M)
                           = cT-2 * cT-1 * cT * sum( A[i,j] * B[j, xt+1] * βT-1(i), j=1,..,M)
                           = cT-2 * cT-1 * cT * βT-2(i)

            scaled β_t(i) = prod(c_j, j=t+1,..,T) * β_t(i)

        Parameters:
        x - observed sequence
        Returns:
        beta - backward probability
        """
        T = x.shape[0]

        beta = np.zeros((T,self.M))
        beta[-1] = 1*scale[-1]
        for t in range(T-2,-1,-1):
            beta[t] = self.A.dot(self.B[:,x[t+1]] * beta[t+1])*scale[t]

        return beta

    def calc_gamma_per_element(self, t, i, alpha, beta, scale):
        """
        calculate probability of state qi at time t given model λ and  observed  sequence x
        γ_t(i) = p(i_t = q_i | x, λ) = α_t(i) * β_t(i) / P(x|λ)

        Note : P(X|λ) = sum(P(α_t(i)), i=1,2,...V, we can save time for calculating this

        scaled version:
            γ_t(i) = α_t(i) * β_t(i) / P(x|λ)
                   = (scaled_α_t(i)/prod(c_j, j=1,..,t)) *  (scaled_ β_t(i)/prod(c_j, j=t+1,..,T)) / P(x|λ)
                   = (scaled_α_t(i) * scaled_β_t(i)) / prod(c_j, j=1,.t,t,+1.,T) * P(x|λ)
                   = (scaled_α_t(i) * scaled_β_t(i)) / ct * prod(c_j, j=1,.t,+1.,T) * P(x|λ)
                   = (scaled_α_t(i) * scaled_β_t(i)) / ct

        so in scaled version, it need divide the scale factor instead of the gamma_denominator

        Parameters:
        x - observed sequence
        alpha - forward probability
        beta - backward probability
        Returns:
        γ_t(i)
        """
        gamma_numerator = alpha[t,i]*beta[t,i]/scale[t]
        return  gamma_numerator
        # gamma_denominator = alpha[-1].sum()
        #
        # return gamma_numerator/gamma_denominator

    def calc_gamma(self, alpha, beta, scale, px=None, updateA=False):
        """
        calculate probability of state qi at time t given model λ and  observed  sequence x
        γ_t(i) = p(i_t = q_i | x, λ) = α_t(i) * β_t(i) / P(x|λ)

        Note : P(X|λ) = sum(P(α_t(i)), i=1,2,...V, we can save time for calculating this

        scaled version:
            γ_t(i) = α_t(i) * β_t(i) / P(x|λ)
                   = (scaled_α_t(i)/prod(c_j, j=1,..,t)) *  (scaled_ β_t(i)/prod(c_j, j=t,..,T)) / P(x|λ)
                   = (scaled_α_t(i) * scaled_β_t(i)) / prod(c_j, j=1,.t,t,+1.,T) * P(x|λ)
                   = (scaled_α_t(i) * scaled_β_t(i)) / ct * prod(c_j, j=1,.t,+1.,T) * P(x|λ)
                   = (scaled_α_t(i) * scaled_β_t(i)) / ct

        so in scaled version, it need divide the scale factor instead of the gamma_denominator

        Parameters:
        x - observed sequence
        alpha - forward probability
        beta - backward probability
        Returns:
        γ
        """
        if updateA:
            gamma_numerator = alpha[:-1] * beta[:-1] / scale[:-1]
        else:
            gamma_numerator = alpha * beta / scale

        # In scaled version, it need divide the scale factor instead of the gamma_denominator
        gamma = np.sum(gamma_numerator, axis=0, keepdims=True).T

        return gamma

    def calc_psai(self, x, alpha, beta, px=None):
        """
        calculate probability of state qi at time t and state qj at time t+1 given model and  observed  sequence x
        ξ_t(i,j) = p(i_t = q_i, i_t+1 = q_j  | x, λ)
                 = α_t(i) * A[i,j] * B[j, xt] * β_t+1(j) / P(x|λ)

        Note : P(X|λ) = sum(P(α_t(i)), i=1,2,...V, we can save time for calculating this

        scaled version:
            ξ_t(i,j) = α_t(i) * A[i,j] * B[j, xt] * β_t+1(j) / P(x|λ)
                     = (scaled_α_t(i)/prod(c_j, j=1,..,t)) * A[i,j] * B[j, xt] * (scaled_ β_t+1(j)/prod(c_j, j=t+1,..,T))
                     = scaled_α_t(i) * A[i,j] * B[j, xt] * scaled_β_t+1(j) / prod(c_j, j=1,..,T)) *  P(x|λ)
                     = scaled_α_t(i) * A[i,j] * B[j, xt] * scaled_β_t+1(j)

        so in scaled version , it dose not need divide neither the scale factor nor the  psai_denominator

        Parameters:
        x - observed sequence
        alpha - forward probability
        beta - backward probability
        Returns:
        ξ
        """
        T = alpha.shape[0]

        psai_numerator = np.zeros((T,self.M,self.M))
        # psai_denominator = alpha[-1].sum()
        for t in range(T-1):
            for i in range(self.M):
                for j in range(self.M):
                    psai_numerator[t,i,j] = alpha[t,i] * self.A[i,j] * self.B[j,x[t+1]] * beta[t+1,j]

        # In scaled version, it dose not need divide neither the scale factor nor the  psai_denominator
        psai = np.sum(psai_numerator, axis=0)

        return psai

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
            alphas, scales, betas = [], [], []
            for i_sample in range(n_samples):
                alpha, scale = self.forward(X[i_sample]) # T * M
                beta = self.backward(X[i_sample], scale) # T * M
                alphas.append(alpha)
                betas.append(beta)
                scales.append(scale)
                # P[i_sample] is P(O|λ) when calculate gamma and psai
                # P(O|λ) = 1 / CT, CT = c1*c2*...*cT
                P[i_sample] = 1 / reduce(lambda x,y:x*y, scale)#
                # P[i_sample] =  np.log(scale).sum()

            # record costs
            costs.append(np.sum(P))

            # Step 2
            # re-estimate pi, A, B

            # Step 2.1 re-estimate pi (mean value)
            self.pi =  np.sum((alphas[i_sample][0] * betas[i_sample][0])/scales[i_sample][0]  \
                                for i_sample in range(n_samples)) / n_samples

            # Step 2.2 re-estimate A, B
            tmp_A, tmp_B = [],[]
            for i_sample in range(n_samples):
                # Step 2.2.1  update A
                A_numerator_psai = self.calc_psai(X[i_sample], alphas[i_sample], betas[i_sample],px=P[i_sample]) # T-1 * M * M
                A_denominator_gamma  = self.calc_gamma(alphas[i_sample],betas[i_sample], scales[i_sample],
                                                       px=P[i_sample], updateA=True) # T-1 * M

                tmp_A.append(A_numerator_psai/A_denominator_gamma)

                # Step 2.2.1  update B
                B_numerator = np.zeros((self.M,self.V)) # M * V
                for j in range(self.M):
                    for k in range(self.V):
                        B_gamma_numerator = 0
                        for t in range(T):
                            if X[i_sample][t] == k:
                                B_gamma_numerator += self.calc_gamma_per_element(t,j,
                                                        alphas[i_sample],betas[i_sample], scales[i_sample])
                        # In scaled version, it dose not need divide the numerator
                        B_numerator[j,k] = B_gamma_numerator#/P[i_sample]

                B_denominator_gamma = self.calc_gamma(alphas[i_sample],betas[i_sample], scales[i_sample],
                                                      px=P[i_sample]) # T * M

                tmp_B.append(B_numerator / B_denominator_gamma) # M*V

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
        states = np.zeros(T,dtype=np.int)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psai[t+1,states[t+1]]

        return states

    def log_likelihood(self,X):
        log_p = []
        for x in X:
            _, scale = self.forward(x)
            p = 1 / reduce(lambda x,y:x*y, scale)
            log_p.append(np.log(p))
        return np.array(log_p)

    def generate(self, length):
        """
        Generate a observed sequence using a trained model

        Parameter:
        length - the length of the observed sequence

        Returns:
        observed_states - the observed sequence
        """
        #

        hidden_states = []
        # generate hidden state sequence
        hidden_states.append(random_sampling(self.pi))
        for i in range(1, length):
            previous_hidden_state = hidden_states[-1]
            hidden_state = random_sampling(self.A[previous_hidden_state])
            hidden_states.append(hidden_state)

        # generate observed state sequence
        observed_states = []
        for i in range(length):
            hidden_state = hidden_states[i]
            observed_state = random_sampling(self.B[hidden_state])
            observed_states.append(observed_state)

        return observed_states