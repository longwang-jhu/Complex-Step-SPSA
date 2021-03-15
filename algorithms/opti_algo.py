#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated: 2021-03-14
@author: Long Wang
"""
import numpy as np

class OptiAlgo(object):
    def __init__(self, a=0, A=0, alpha=0.602, c=0, gamma=0.101,
                 iter_num=1, dir_num=1, rep_num=1,
                 theta_0=None, loss_true=None, loss_noisy=None,
                 record_theta_flag=False, record_loss_flag=False,
                 seed=1):

        # step size: a_k = a / (k+1+A) ** alpha
        # perturbation size: c_k = c / (k+1) ** gamma
        # dir_num: number of random directions per iteration
        np.random.seed(seed)

        self.a = a
        self.A = A
        self.alpha = alpha
        self.c = c
        self.gamma = gamma

        self.iter_num = iter_num
        self.dir_num = dir_num
        self.rep_num = rep_num

        self.theta_0 = theta_0
        self.loss_true = loss_true
        self.loss_noisy = loss_noisy

        self.record_theta_flag = record_theta_flag
        self.record_loss_flag = record_loss_flag

        self.p = theta_0.shape[0]
        if self.record_theta_flag:
            self.theta_ks = np.zeros((self.p, self.iter_num, self.rep_num))
        if self.record_loss_flag:
            self.loss_ks = np.zeros((self.iter_num, self.rep_num))

    def train(self):
        pass

    def record_result(self, iter_idx=0, rep_idx=0, theta_k=None):
        if self.record_theta_flag:
            self.theta_ks[:,iter_idx,rep_idx] = theta_k
        if self.record_loss_flag:
            self.loss_ks[iter_idx,rep_idx] = self.loss_true(theta_k)

    def show_result(self, iter_idx, rep_idx):
        if self.record_loss_flag and divmod(iter_idx+1, 100)[1] == 0:
            print("iter:", iter_idx+1, "loss:", self.loss_ks[iter_idx,rep_idx])