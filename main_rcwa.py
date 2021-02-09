import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils import data_utils
from utils import calc_utils

# ================= Unit Define
meters = 1
centimeters = 1e-2 * meters
millimeters = 1e-3 * meters

# ================= Constant Define
c0 = 3e8
e0 = 8.85e-12
u0 = 1.256e-6
yeta0 = np.sqrt(u0/e0)
# print('yeta0',yeta0)

path_gold = 'Au3-Drude.txt'
eps_gold_file = data_utils.load_property_txt(path_gold,2743,3636)
# print('eps_gold',eps_gold.shape)
eps_gold = eps_gold_file[:,1] + eps_gold_file[:,2]*1j
# print('eps_gold',eps_gold)

path_SiNx = 'SiNx_property.mat'
eps_SiNx = data_utils.load_property_mat(path_SiNx)
# print('eps_SiNx REAL',eps_SiNx['eps_SiNx_real'])
# print('eps_SiNx IMAG',eps_SiNx['eps_SiNx_imag'])
eps_SiNx = eps_SiNx['eps_SiNx_real'] + eps_SiNx['eps_SiNx_imag']*1j
# print('eps_SiNx',eps_SiNx.shape)

freq = eps_gold_file[:,0]*1e12
# print(len(freq))
# print(freq.shape[0])

# plot property
n_SiNx = np.sqrt(eps_SiNx)
# print(n_SiNx)
# plt.plot(freq,np.real(n_SiNx))
# plt.plot(freq,np.imag(n_SiNx))
# plt.show()


# ================= Calculation Start
for i_freq in range(len(freq)):
    lam0 = c0/freq[i_freq]
    # print(lam0)
    ginc = np.array([[0,0,1]]).T
    # print(ginc)
    EP = np.array([[0, 1, 0]]).T

    # === Device Params
    ur1 = 1.  # permeability in reflection region
    er1 = 1.  # permeability in reflection region
    ur2 = 1.  # permeability in transmission region
    er2 = 1.  # permeability in transmission region
    urd = 1.  # permeability of device
    erd = np.conjugate(eps_SiNx[i_freq])  # permeability of device

    Lx = 0.005 * millimeters  # period along x
    Ly = 0.005 * millimeters  # period along y
    d1 = 0.00015 * millimeters  # thickness of layer 1
    d2 = 0.0005 * millimeters  # thickness of layer 2
    d3 = 0.00015 * millimeters  # thickness of layer 3
    w = 0.52 * Ly  # length of one side of square

    # === RCWA Params
    Nx = 512
    Ny = np.round(Nx*Ly/Lx).astype(int)
    PQ = 1 * np.array([1, 31])

    # === Define Structure in Layers
    nxc = np.floor(Nx/2)
    nyc = np.floor(Ny/2)
    ER1 = 1 * np.ones((Nx,Ny))
    ER2 = erd * np.ones((Nx,Ny))
    ER3 = np.conjugate(eps_gold[i_freq]) * np.ones((Nx,Ny))
    ER = np.concatenate([ER1[...,np.newaxis],ER2[...,np.newaxis],ER3[...,np.newaxis]],axis=-1)  # [512,512,3]
    # print(ER)
    # print(ER.shape)
    UR1 = urd * np.ones((Nx,Ny))
    UR2 = urd * np.ones((Nx,Ny))
    UR3 = urd * np.ones((Nx,Ny))
    UR = np.concatenate([UR1[...,np.newaxis],UR2[...,np.newaxis],UR3[...,np.newaxis]],axis=-1)  # [512,512,3]
    print(UR.shape)
    L = np.array([[d1,d2,d3]])

    # === Cross Sectional Grid
    dx = Lx/Nx  # grid resolution along x
    dy = Ly/Ny  # grid resolution along y
    xa = np.arange(Nx) * dx  # x axis array
    xa = xa - np.mean(xa)  # center x axis at zero
    ya = np.arange(Ny) * dy  # y axis array
    ya = ya - np.mean(ya)  # center y axis at zero
    x_axis, y_axis = np.meshgrid(xa,ya)

    ny1 = np.round(nxc-((w/Ly)*Nx)/2).astype(int)
    ny2 = np.round(nxc+((w/Ly)*Nx)/2).astype(int)
    ER[ny1-1:ny2,ny1-1:ny2,0] = np.conjugate(eps_gold[i_freq])
    mm, nn, ll = ER.shape
    # print(PQ[0])
    for i_ll in range(ll):
        URC_i = calc_utils.convmat(UR[...,i_ll][...,np.newaxis], PQ[0], PQ[1])
        # print('URC_i',URC_i)
        # print('URC_i', URC_i.shape)
        ERC_i = calc_utils.convmat(ER[..., i_ll][..., np.newaxis], PQ[0], PQ[1])
        # print('ERC_i',ERC_i)
        # print('ERC_i', ERC_i.shape)
        if i_ll == 0:
            URC = URC_i[..., np.newaxis]
            ERC = ERC_i[..., np.newaxis]
        else:
            URC = np.concatenate((URC, URC_i[..., np.newaxis]), axis=-1)
            ERC = np.concatenate((ERC, ERC_i[..., np.newaxis]), axis=-1)

    print(URC.shape)
    print(ERC.shape)

    # === Wave Vector Expansion

    pass