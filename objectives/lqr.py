#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated: 2021-03-14
@author: Long Wang
"""

import numpy as np
import control

class LQR:
    def __init__(self, p=2, T=100, x_0=None):
        if p == 2:
            A = np.array([[1,1],[1,0]])
            B = np.array([[0],[1]])
            self.n = 2
            self.m = 1
        elif p == 12:
            A = np.array([[-2.5, 1.2, 4.3, 0.1],
                          [0.97, -10.3, 0.4, -6.1],
                          [-9.2, 1.1, -4.9, 0.3],
                          [1.1, 0.9, -3.4, -0.9]])
            B = np.array([[1.1, 0.4, -0.2],
                          [-3.2, 1.4, 0.0],
                          [-0.8, 0.1, 3.0],
                          [-1.1, -0.9, 5.2]])
            self.n = 4
            self.m = 3

        C = np.eye(A.shape[0])
        D = np.zeros((A.shape[0], B.shape[1]))
        sysd = control.ss(A, B, C, D).sample(0.1)
        self.A, self.B, _, _ = control.ssdata(sysd)

        self.Q = np.eye(self.n)
        self.R = np.eye(self.m)
        self.T = T
        self.x_0 = x_0
        self.w_Sigma = 0.5 * np.eye(self.n)
        self.v_Sigma = 0.5 * np.eye(self.n)

    def next_state(self, x, u):
        x_next = np.dot(self.A, x) + np.dot(self.B, u)
        w = np.random.multivariate_normal(np.zeros(self.n), self.w_Sigma).reshape((self.n,1))
        return x_next + w

    def measurement_state(self, x):
        v = np.random.multivariate_normal(np.zeros(self.n), self.v_Sigma).reshape((self.n,1))
        return x + v

    def compute_cost(self, K): # assume K.shape = (p,_)
        K_mat = K.reshape((self.m, self.n))

        cost_avg = 0
        cost_n = 10
        for i in range(cost_n):
            cost = 0
            x_t = self.x_0.copy()
            for t in range(self.T):
                cost += np.dot(x_t.T, np.dot(self.Q, x_t))
                u_t = -np.dot(K_mat, x_t)
                cost += np.dot(u_t.T, np.dot(self.R, u_t))
                x_t = self.next_state(x_t, u_t)
            cost += np.dot(np.dot(x_t.T, self.Q), x_t)
            cost_avg += cost[0][0] / cost_n
        return cost_avg

    def compute_cost_noisy(self, K):
        K_mat = K.reshape((self.m, self.n))

        if type(K_mat[0][0]) == np.complex128:
            cost = 0 + 0j
        else:
            cost = 0

        x_t = self.x_0.copy()
        y_t = self.measurement_state(x_t)
        for t in range(self.T):
            cost += np.dot(y_t.T, np.dot(self.Q, y_t))
            u_t = -np.dot(K_mat, x_t)
            cost += np.dot(u_t.T, np.dot(self.R, u_t))
            x_t = self.next_state(x_t, u_t)
            y_t = self.measurement_state(x_t)
        cost += np.dot(np.dot(y_t.T, self.Q), y_t)
        return cost[0][0]

if __name__ == "__main__":
    # p = 2; n = 2; m = 1
    # x_0 = np.random.uniform(-1,1,(n,1))

    p = 12; n = 4; m = 3
    # x_0 = np.array([[0, 0]])
    x_0 = 20 * np.array([1, 2, -1, -0.5]).reshape(n,1)
    # x_0 = np.random.uniform(-20,20,(n,1))

    LQR_model = LQR(p=p, T=100, x_0=x_0)

    # K = np.ones((m,n))
    # K_star = np.array([[-6.4641, -3.7321]])
    # print(LQR_model.compute_cost(x_0, K_star))

    # K_lqr, _, _ = control.lqr(LQR_model.A, LQR_model.B, LQR_model.Q, LQR_model.R)
    # K_lqr = np.array([[6.46410162, 3.73205081]])
    K_lqr = np.array([
        [1.60233232e-01, -1.36227805e-01, -9.93576677e-02, -4.28244630e-02],
        [7.47596033e-02,  9.05753832e-02,  7.46951286e-02, -1.53947620e-01],
        [3.65372978e-01, -2.59862175e-04,  5.91522023e-02, 8.25660846e-01]])

    C_lqr = LQR_model.compute_cost(K_lqr.flatten())
    print(C_lqr) # 4149.38952236

    C_lqr_noisy =LQR_model.compute_cost_noisy(K_lqr.flatten())
    print(C_lqr_noisy) # 4149.38952236

    K_lqr_plus = K_lqr + np.ones(K_lqr.shape) * 0.1j
    C_lqr_plus = LQR_model.compute_cost(K_lqr_plus)
    print(C_lqr_plus)