#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated: 2021-03-14
@author: Long Wang
"""
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
from utility import norm_error

# import algorithms
from algorithms.spsa import SPSA
from algorithms.cs_spsa import CsSPSA

# import objective
from objectives.lqr import LQR

###
today = date.today()
np.random.seed(100)

p = 12; T = 100
n = 4; m = 3
x_0 = 20 * np.array([1, 2, -1, -0.5]).reshape(n,1)

LQR_model = LQR(p=p, T=T, x_0=x_0)
K_star = np.array([
    [1.60233232e-01, -1.36227805e-01, -9.93576677e-02, -4.28244630e-02],
    [7.47596033e-02,  9.05753832e-02,  7.46951286e-02, -1.53947620e-01],
    [3.65372978e-01, -2.59862175e-04,  5.91522023e-02, 8.25660846e-01]])
theta_star = K_star.flatten()
# loss_star = loss_true(theta_star)
loss_star = 4149.38952236

def loss_true(theta):
    return LQR_model.compute_cost(theta)

def loss_noisy(theta):
    return LQR_model.compute_cost_noisy(theta)

# inital value
K_0 = np.ones(K_star.shape) * 2
theta_0 = K_0.flatten()
loss_0 = loss_true(theta_0)
print('loss_0:', loss_0)

# parameters
alpha = 0.668; gamma = 0.167
iter_num = 500; rep_num = 5

print('running SPSA')
SPSA_solver = SPSA(a=0.00005, c=0.5, A=100, alpha=alpha, gamma=gamma,
                    iter_num=iter_num, rep_num=rep_num,
                    theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy,
                    record_loss_flag=True)
# SPSA_solver.train()
SPSA_loss_error = norm_error.get_norm_loss_error(SPSA_solver.loss_ks, loss_0, loss_star)
plt.yscale('log')
plt.plot(SPSA_loss_error, 'k-')

print('running CsSPSA')
CsSPSA_solver = CsSPSA(a=0.0001, c=0.5, A=100, alpha=alpha, gamma=gamma,
                       iter_num=iter_num, rep_num=rep_num,
                       theta_0=theta_0, loss_true=loss_true, loss_noisy=loss_noisy,
                       record_loss_flag=True)
CsSPSA_solver.train()
CsSPSA_loss_error = norm_error.get_norm_loss_error(CsSPSA_solver.loss_ks, loss_0, loss_star)
plt.yscale('log')
plt.plot(CsSPSA_loss_error, 'k-')


# plot loss
fig = plt.figure()
ax = fig.add_subplot()
plt.grid()

line_SPSA, = ax.plot(SPSA_loss_error, 'k--', dashes=(5,5))
line_SPSA_for_legend, = ax.plot([1], 'k--')
line_CS_SPSA, = ax.plot(CsSPSA_loss_error, 'k-')

plt.xlim(xmin=0, xmax=iter_num)
plt.yscale("log")
plt.legend((line_SPSA_for_legend, line_CS_SPSA),
           ("SPSA", "CS-SPSA"), loc="best")
plt.xlabel("Number of Iterations")
plt.ylabel("Normalized Errors in Loss (log scale)")
plt.savefig("figures/LQR-loss-" + str(today) + ".pdf")#, bbox_inches='tight')
plt.show()