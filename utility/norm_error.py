#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 17:26:59 2021

@author: Long Wang
"""
import numpy as np

def get_norm_loss_error(loss_ks, loss_0, loss_star, multi=1):
    loss_error = (np.mean(loss_ks, axis=1) - loss_star) / (loss_0 - loss_star)
    loss_error = list(np.repeat(loss_error, multi))
    loss_error.insert(0, 1)
    return loss_error

def get_norm_theta_error(theta_ks, theta_0, theta_star, multi=1):
    theta_error = np.linalg.norm(theta_ks - theta_star[:,None,None], axis=0)
    theta_error = np.mean(theta_error, axis=1) / np.linalg.norm(theta_0 - theta_star)
    theta_error = list(np.repeat(theta_error, multi))
    theta_error.insert(0, 1)
    return theta_error