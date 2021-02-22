import numpy as np
from numpy import linalg as LA
from scipy import linalg as SLA
import cmath
import matplotlib
import matplotlib.pyplot as plt

from utils import data_utils
from utils import calc_utils
from utils import rcwa_utils

pi = np.pi

# ================= Unit Define
meters = 1
centimeters = 1e-2 * meters
millimeters = 1e-3 * meters


# ================= Constant Define
c0 = 3e8
e0 = 8.85e-12
u0 = 1.256e-6
yeta0 = np.sqrt(u0/e0)

path_gold = 'Au3-Drude.txt'
eps_gold_file = data_utils.load_property_txt(path_gold,2743,3636)
eps_gold = eps_gold_file[:,1] + eps_gold_file[:,2]*1j

path_SiNx = 'SiNx_property.mat'
eps_SiNx = data_utils.load_property_mat(path_SiNx)
eps_SiNx = eps_SiNx['eps_SiNx_real'] + eps_SiNx['eps_SiNx_imag']*1j

freq = eps_gold_file[:,0]*1e12

# plot property
n_SiNx = np.sqrt(eps_SiNx)
# print(n_SiNx)
# plt.plot(freq,np.real(n_SiNx))
# plt.plot(freq,np.imag(n_SiNx))
# plt.show()

# ================= RCWA Solver
R_total, T_total = rcwa_utils.rcwa_solver(freq, eps_gold, eps_SiNx)


# ================= Spectra Plot
plt.figure(1)
plt.plot(freq, R_total)
plt.figure(2)
plt.plot(freq, T_total)
plt.show()

path = 'fRT.npz'
np.savez(path, freq=freq, R=R_total, T=T_total)
print('FILE SAVED')
pass