#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated: 2021-03-14
@author: Long Wang
"""
from datetime import date
import numpy as np
import matplotlib.pyplot as plt

from objectives.Gaussian_signal_noise import GaussianSignalNoise

# from algorithms.spsa_online import SPSAOnline
from algorithms.cs_spsa_online import CsSPSAOnline

from utility import norm_error

np.random.seed(99)

# initialization
p = 10; n = 100
loss_obj = GaussianSignalNoise(p=p, n=n)
loss_star = loss_obj.get_loss_true_mu_Sigma(loss_obj.mu, loss_obj.Sigma)
print('loss_star:', loss_star)

Sigma_0 = np.eye(p)
theta_0 = Sigma_0[np.tril_indices(p)]
theta_0 += np.random.rand(theta_0.size)
Sigma_0 = loss_obj.convert_theta_to_Sigma(theta_0)

Sigma_0_norm_error = np.linalg.norm(Sigma_0 - loss_obj.Sigma, ord="fro")
loss_0 = loss_obj.get_loss_true(theta_0)
print('loss_0:', loss_0)
print()

# optimizer parameters
alpha = 0.602
gamma = 0.151
iter_num = 1000; rep_num = 1

### SGD ###
print('Running SGD')
a_s = [0.05, 0.025, 0.01]
SGD_loss_ks = np.empty((iter_num, rep_num, 3))
for a_idx in range(3):
    a = a_s[a_idx]
    for rep_idx in range(rep_num):
        print("Running rep_idx:", rep_idx + 1, "/", rep_num)
        Sigma_k = Sigma_0.copy()
        for iter_idx in range(iter_num):
            a_k = a / (iter_idx + 1 + 100) ** alpha
            grad_Sigma_k = loss_obj.get_grad_Sigma_noisy(iter_idx, loss_obj.mu, Sigma_k)
            Sigma_k -= a_k * grad_Sigma_k

            SGD_loss_ks[iter_idx, rep_idx, a_idx] = loss_obj.get_loss_true_mu_Sigma(loss_obj.mu, Sigma_k)
            # if (iter_idx + 1) % 100 == 0:
            #     print('Iter:', iter_idx + 1, 'Loss:', SGD_loss_ks[iter_idx, rep_idx])

SGD_loss_error_1 = norm_error.get_norm_loss_error(SGD_loss_ks[:,:,0], loss_0, loss_star)
SGD_loss_error_2 = norm_error.get_norm_loss_error(SGD_loss_ks[:,:,1], loss_0, loss_star)
SGD_loss_error_3 = norm_error.get_norm_loss_error(SGD_loss_ks[:,:,2], loss_0, loss_star)

# plt.figure(); plt.grid()
# plt.plot(SGD_loss_error)
# plt.show()
# print('Terminal loss:', SGD_loss_error[-1])

### SPSA ###
# print("running SPSA")
# SPSA_optimizer = SPSAOnline(a=0.05, A=100, alpha=alpha,
#                             c=0.05, gamma=gamma,
#                             iter_num=iter_num, rep_num=rep_num,
#                             theta_0=theta_0, loss_obj=loss_obj,
#                             record_loss_flag=True)

# SPSA_optimizer.train()
# SPSA_loss_error = norm_error.get_norm_loss_error(SPSA_optimizer.loss_ks, loss_0, loss_star)
# plt.figure(); plt.grid()
# plt.plot(SPSA_loss_error)
# print('Terminal loss:', SPSA_loss_error[-1])

### CS-SPSA ###
print('Running CS-SPSA')
CsSPSA_optimizer = CsSPSAOnline(a=0.05, A=100, alpha=alpha,
                                c=0.05, gamma=gamma,
                                iter_num=iter_num, rep_num=rep_num,
                                theta_0=theta_0, loss_obj=loss_obj,
                                record_loss_flag=True)

CsSPSA_optimizer.train()
CsSPSA_loss_error = norm_error.get_norm_loss_error(CsSPSA_optimizer.loss_ks, loss_0, loss_star)
plt.figure(); plt.grid()
plt.plot(CsSPSA_loss_error)
plt.show()
print('Terminal loss:', CsSPSA_loss_error[-1])

### plot ###
today = date.today()

# plot loss
plt.figure(); plt.grid()
plt.plot(SGD_loss_error_1)
plt.plot(SGD_loss_error_2)
plt.plot(SGD_loss_error_3)
plt.plot(CsSPSA_loss_error)

plt.xlim(xmin=0, xmax=iter_num)
plt.yscale("log")
plt.legend(("SGD (a=0.05)", "SGD (a=0.025)", "SGD (a=0.01)", "CS-SPSA"), loc="best")
plt.xlabel("Number of Iterations")
plt.ylabel("Normalized Errors in Loss (log scale)")
plt.savefig("figures/Gaussian_signal_noise_p_" + str(p) + "_loss_" + str(today) + ".pdf", bbox_inches='tight')
plt.show()

print("Finished")