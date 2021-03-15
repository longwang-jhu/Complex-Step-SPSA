#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 19:31:45 2020

@author: Long Wang
"""

import numpy as np
from algorithms.opti_algo import OptiAlgo

class SPSA(OptiAlgo):
    def get_delta_all(self):
        self.delta_all = np.round(np.random.rand(self.p, self.dir_num, self.iter_num, self.rep_num)) * 2 - 1

    def get_grad_est(self, iter_idx=0, rep_idx=0, theta_k=None):
        c_k = self.c / (iter_idx + 1) ** self.gamma
        grad_k = np.zeros(self.p)
        for dir_idx in range(self.dir_num):
            delta_k = self.delta_all[:, dir_idx, iter_idx, rep_idx]
            loss_plus = self.loss_noisy(theta_k + c_k * delta_k)
            loss_minus = self.loss_noisy(theta_k - c_k * delta_k)
            grad_k += (loss_plus - loss_minus) / (2 * c_k) * delta_k
        return grad_k / self.dir_num

    def train(self):
        self.get_delta_all()
        for rep_idx in range(self.rep_num):
            print("running rep_idx:", rep_idx + 1, "/", self.rep_num)
            theta_k = self.theta_0.copy() # reset theta_k
            for iter_idx in range(self.iter_num):
                a_k = self.a / (iter_idx + 1 + self.A) ** self.alpha
                g_k = self.get_grad_est(iter_idx, rep_idx, theta_k)
                theta_k -= a_k * g_k

                # record result
                self.record_result(iter_idx, rep_idx, theta_k)
                # show result
                self.show_result(iter_idx, rep_idx)