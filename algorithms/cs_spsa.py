#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 21:35:30 2020

@author: Long Wan
"""

import numpy as np
from algorithms.spsa import SPSA

class CsSPSA(SPSA):
    def get_grad_est(self, iter_idx=0, rep_idx=0, theta_k=None):
        c_k = self.c / (iter_idx + 1) ** self.gamma
        grad_k = np.zeros(self.p)
        for dir_idx in range(self.dir_num):
            delta_k = self.delta_all[:,dir_idx, iter_idx, rep_idx]
            theta_k_plus = np.array(theta_k, dtype = complex) + 1j * c_k * delta_k
            loss_plus = self.loss_noisy(theta_k_plus)
            grad_k += loss_plus.imag / c_k * delta_k
        return grad_k / self.dir_num