#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated: 2021-03-14
@author: Long Wang
"""

import numpy as np
from algorithms.spsa import SPSA

class SPSAOnline(SPSA):
    def get_grad_est(self, iter_idx=0, rep_idx=0, theta_k=None):
        c_k = self.c / (iter_idx + 1) ** self.gamma
        grad_k = np.zeros(self.p)
        for dir_idx in range(self.dir_num):
            delta_k = self.delta_ks[:, dir_idx, iter_idx, rep_idx]
            loss_plus = self.loss_obj.get_loss_noisy(iter_idx, theta_k + c_k * delta_k)
            loss_minus = self.loss_obj.get_loss_noisy(iter_idx, theta_k - c_k * delta_k)
            grad_k += (loss_plus - loss_minus) / (2 * c_k) * delta_k
        return grad_k / self.dir_num