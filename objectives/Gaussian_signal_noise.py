#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated: 2021-03-14
@author: Long Wang
"""

import numpy as np
from scipy.stats import multivariate_normal

class GaussianSignalNoise:
    def __init__(self, p=4, n=100,
                 known_mu_flag=True, known_Sigma_flag=False):
        np.random.seed(99)

        self.p = p
        self.n = n

        self.known_mu_flag = known_mu_flag
        self.known_Sigma_flag = known_Sigma_flag

        self.mu = np.zeros(self.p)
        self.Sigma = np.eye(self.p)

        # generate noise
        self.L = np.zeros([self.n, self.p, self.p])
        self.P = np.zeros([self.n, self.p, self.p])
        for i in range(self.n):
            self.L[i] = np.tril(np.random.rand(self.p, self.p)) * 0.1
            self.P[i] = np.dot(self.L[i], self.L[i].T) * np.sqrt(i)

        # generate data
        self.x = np.empty([self.n, self.p])
        for i in range(self.n):
            self.x[i] = np.random.multivariate_normal(self.mu, self.Sigma + self.P[i])

    # convert theta to mu and Sigma
    def convert_theta_to_Sigma(self, theta):
        idx_lower = np.tril_indices(self.p)
        L = np.zeros([self.p, self.p])
        L[idx_lower] = theta
        return np.dot(L, L.T)
    def convert_theta_to_Sigma_complex(self, theta_complex):
        idx_lower = np.tril_indices(self.p)
        L = np.zeros([self.p, self.p], dtype="complex128")
        L[idx_lower] = theta_complex
        return np.dot(L, L.T)

    def convert_theta_to_mu_Sigma(self, theta):
        if (not self.known_mu_flag) and (not self.known_Sigma_flag):
            mu = theta[:self.p]
            Sigma = self.convert_theta_to_Sigma(theta[self.p:])
        elif self.known_mu_flag and (not self.known_Sigma_flag):
            mu = self.mu
            Sigma = self.convert_theta_to_Sigma(theta)
        return mu, Sigma
    def convert_theta_to_mu_Sigma_complex(self, theta):
        if (not self.known_mu_flag) and (not self.known_Sigma_flag):
            mu_complex = theta[:self.p]
            Sigma_complex = self.convert_theta_to_Sigma_complex(theta[self.p:])
        elif self.known_mu_flag and (not self.known_Sigma_flag):
            mu_complex = self.mu + 0j
            Sigma_complex = self.convert_theta_to_Sigma_complex(theta)
        return mu_complex, Sigma_complex

    # get logpdf
    def get_logpdf(self, sample_idx, mu, Sigma):
        log2pi = np.log(2 * np.pi)
        logdet = np.log(np.linalg.det(Sigma + self.P[sample_idx]))
        maha_dist = sum((self.x[sample_idx] - mu)
                        * (np.linalg.solve(Sigma + self.P[sample_idx], self.x[sample_idx] - mu)))
        return -1/2 * (self.p * log2pi + logdet + maha_dist)
    def get_logpdf_complex(self, sample_idx, mu_complex, Sigma_complex):
        log2pi = np.log(2 * np.pi)
        logdet = np.log(np.linalg.det(Sigma_complex + self.P[sample_idx]))
        maha_dist = sum((self.x[sample_idx] - mu_complex)
                        * (np.linalg.solve(Sigma_complex + self.P[sample_idx], self.x[sample_idx] - mu_complex)))
        return -1/2 * (self.p * log2pi + logdet + maha_dist)

    # get noisy loss value
    def get_loss_noisy(self, iter_idx, theta):
        mu, Sigma = self.convert_theta_to_mu_Sigma(theta)
        sample_idx = np.remainder(iter_idx, self.n)
        logpdf = np.log(multivariate_normal.pdf(self.x[sample_idx],
                                                mean=mu, cov=Sigma + self.P[sample_idx]))
        return -logpdf

    def get_loss_noisy_complex(self, iter_idx, theta):
        mu_complex, Sigma_complex = self.convert_theta_to_mu_Sigma_complex(theta)
        sample_idx = np.remainder(iter_idx, self.n)
        logpdf = self.get_logpdf_complex(sample_idx, mu_complex, Sigma_complex + self.P[sample_idx])
        return -logpdf

    # get true loss value
    def get_loss_true_mu_Sigma(self, mu, Sigma):
        logpdf_all = 0
        for sample_idx in range(self.n):
            logpdf_all += np.log(multivariate_normal.pdf(self.x[sample_idx],
                                                         mean=mu, cov=Sigma + self.P[sample_idx]))
        return -logpdf_all

    def get_loss_true(self, theta):
        mu, Sigma = self.convert_theta_to_mu_Sigma(theta)
        return self.get_loss_true_mu_Sigma(mu, Sigma)

    # get noisy gradient w.r.t. Sigma
    def get_grad_Sigma_noisy(self, iter_idx, mu, Sigma):
        sample_idx = np.remainder(iter_idx, self.n)

        W = np.ones((self.p, self.p)) - np.eye(self.p) / 2
        Sigma_plus_P_inv = np.linalg.inv(Sigma + self.P[sample_idx])
        mean_outer_prod = np.outer(self.x[sample_idx] - mu, self.x[sample_idx] - mu)
        grad_Sigma_noisy = Sigma_plus_P_inv + np.dot(np.dot(Sigma_plus_P_inv, mean_outer_prod), Sigma_plus_P_inv)
        grad_Sigma_noisy = W * grad_Sigma_noisy

        return -grad_Sigma_noisy


if __name__ == "__main__":
    import time
    np.random.seed(99)

    p = 100
    loss_obj = GaussianSignalNoise(p)
    print("loss_star:", loss_obj.get_loss_true_mu_Sigma(loss_obj.mu, loss_obj.Sigma))

    Sigma_0 = np.eye(p)
    theta_0 = Sigma_0[np.tril_indices(p)]
    theta_0 += np.random.rand(theta_0.size) * 0.5
    Sigma_0 = loss_obj.convert_theta_to_Sigma(theta_0)
    print("loss_0:", loss_obj.get_loss_true(theta_0))
    print()

    ### compare running time ###
    n_rep = 1000
    start_SGD = time.time()
    for j in range(n_rep):
        for i in range(loss_obj.n):
            loss_obj.get_grad_Sigma_noisy(i,loss_obj.mu, Sigma_0)
    end_SGD = time.time()
    print("SGD time:", end_SGD - start_SGD)

    start_CS_SPSA = time.time()
    for j in range(n_rep):
        for i in range(loss_obj.n):
            loss_obj.get_logpdf_complex(i, loss_obj.mu + 0j, Sigma_0 + 0j + loss_obj.P[i])
    end_CS_SPSA = time.time()
    print("CS-SPSA time:", end_CS_SPSA - start_CS_SPSA)

    print((end_SGD - start_SGD) / (end_CS_SPSA - start_CS_SPSA))