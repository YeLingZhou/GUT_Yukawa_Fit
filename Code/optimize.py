# ======== The code for numerical scanning in arXiv:2511.16196  ========
# =====  Authors: Zi-Qiang Chen, Gao-Xiang Fang and Ye-Ling Zhou   =====

import numpy as np
import random
from mpi4py import MPI
import argparse
import time
import os
from itertools import product
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

# ==================== Input parameters ====================
v = 174 * 10**9  # GeV

# Quark yukawa
ubf = 2.54 * 10**-6
cbf = 1.37 * 10**-3
tbf = 0.428
uer = 0.86 * 10**-6
cer = 0.04 * 10**-3
ter = 0.003

dbf = 6.56 * 10**-6
sbf = 1.24 * 10**-4
bbf = 5.7 * 10**-3
der = 0.65 * 10**-6
ser = 0.06 * 10**-4
ber = 0.05 * 10**-3

# Lepton yukawa 
ebf = 2.70341 * 10**-6
mubf = 5.70705 * 10**-4
taubf = 9.70200 * 10**-3
eer = 0.0270 * 10**-6
muer = 0.0570 * 10**-4
tauer = 0.0970 * 10**-3

# Quark mixing
q_theta12 = 0.22739
q_theta23 = 4.858 * 10**-2
q_theta13 = 4.202 * 10**-3
q_delta_cp = 1.207
q_theta12_er = 0.0006
q_theta23_er = 0.06 * 10**-2
q_theta13_er = 0.13 * 10**-3
q_delta_cp_er = 0.054

# ==================== Neutrino Parameters for NO ====================
def setup_neutrino_params(version='1st'):
    """
    version: '1st' or '2nd'
    """
    if version == '1st':
        # NO \theta_{23} in 1Oct
        delta_ms21 = 7.498 * 10**-5
        delta_ms31 = 2.513 * 10**-3
        delta_ms21_er = 0.116 * 10**-5
        delta_ms31_er = 0.5 * (0.019 + 0.021) * 10**-3

        v_theta12 = 0.309 
        v_theta23 = 0.470
        v_theta13 = 0.02215 
        v_theta12_er = 0.009
        v_theta23_er = 0.5 * (0.013 + 0.017) 
        v_theta13_er = 0.5 * (0.00056 + 0.00058)
        
    elif version == '2nd':
        # NO \theta_{23} in 2Oct
        delta_ms21 = 7.498 * 10**-5
        delta_ms31 = 2.534 * 10**-3
        delta_ms21_er = 0.116 * 10**-5
        delta_ms31_er = 0.5 * (0.023 + 0.025) * 10**-3

        v_theta12 = 0.309 
        v_theta23 = 0.561
        v_theta13 = 0.02195 
        v_theta12_er = 0.009 
        v_theta23_er = 0.5 * (0.012 + 0.015) 
        v_theta13_er = 0.5 * (0.00054 + 0.00058)
    
    else:
        raise ValueError("version must be '1st' or '2nd'")
    
    # Return as a dictionary
    return {
        'delta_ms21': delta_ms21,
        'delta_ms31': delta_ms31,
        'delta_ms21_er': delta_ms21_er,
        'delta_ms31_er': delta_ms31_er,
        'v_theta12': v_theta12,
        'v_theta13': v_theta13,
        'v_theta23': v_theta23,
        'v_theta12_er': v_theta12_er,
        'v_theta13_er': v_theta13_er,
        'v_theta23_er': v_theta23_er
    }

# ==================== \chi^2 functions for 3 Models ====================

def Chi2_M1(d11, d22, d33, s11, s12, s13, s22, s23, s33, s1, s2, s3, s4, s5, s6, r1, r2, omega, m0, neutrino_params):

    delta_ms21 = neutrino_params['delta_ms21']
    delta_ms31 = neutrino_params['delta_ms31']
    delta_ms21_er = neutrino_params['delta_ms21_er']
    delta_ms31_er = neutrino_params['delta_ms31_er']
    v_theta12 = neutrino_params['v_theta12']
    v_theta13 = neutrino_params['v_theta13']
    v_theta23 = neutrino_params['v_theta23']
    v_theta12_er = neutrino_params['v_theta12_er']
    v_theta13_er = neutrino_params['v_theta13_er']
    v_theta23_er = neutrino_params['v_theta23_er']
    
    I = 1j
    
    # Diagonal matrix D
    D = np.array([[d11, 0, 0],
                  [0,  d22, 0],
                  [0, 0, d33]], dtype=complex)
    
    # Symmetric matric S
    S = np.array([[s11 * np.exp(I*s1), s12 * np.exp(I*s2), s13 * np.exp(I*s3)],
                  [s12 * np.exp(I*s2), s22 * np.exp(I*s4), s23 * np.exp(I*s5)],
                  [s13 * np.exp(I*s3), s23 * np.exp(I*s5), s33 * np.exp(I*s6)]], dtype=complex)
    
    # Yukawa matrix in M1
    yu = S + D
    yd = r2 * S + np.exp(I * omega) * r1 * D 
    ye = r2 * S - 3 * np.exp(I * omega) * r1 * D
    yv = S - 3 * D
    
    # Neutrino mass matrix given by Type-I Seesaw
    mv = - m0 * yv @ np.linalg.inv(D) @ yv.T
    
    # Computing eigenvalues and eigenvectors 
    # Up-type quarks
    yu_squared = yu @ yu.conj().T
    u2, U = np.linalg.eig(yu_squared)
    u2_sorted_indices = np.argsort(u2.real)
    u2_sorted = u2[u2_sorted_indices]
    U_sorted = U[:, u2_sorted_indices]
    U1 = U_sorted[:, 0] / np.linalg.norm(U_sorted[:, 0])
    U2 = U_sorted[:, 1] / np.linalg.norm(U_sorted[:, 1]) 
    U3 = U_sorted[:, 2] / np.linalg.norm(U_sorted[:, 2])
    uprs, cprs, tprs = u2_sorted[0].real, u2_sorted[1].real, u2_sorted[2].real
    
    # Down-type quarks 
    yd_squared = yd @ yd.conj().T
    d2, D = np.linalg.eig(yd_squared)
    d2_sorted_indices = np.argsort(d2.real)
    d2_sorted = d2[d2_sorted_indices]
    D_sorted = D[:, d2_sorted_indices]
    D1 = D_sorted[:, 0] / np.linalg.norm(D_sorted[:, 0])
    D2 = D_sorted[:, 1] / np.linalg.norm(D_sorted[:, 1])
    D3 = D_sorted[:, 2] / np.linalg.norm(D_sorted[:, 2])
    dprs, sprs, bprs = d2_sorted[0].real, d2_sorted[1].real, d2_sorted[2].real
    
    # Charged leptons
    ye_squared = ye @ ye.conj().T
    l2, E = np.linalg.eig(ye_squared)
    l2_sorted_indices = np.argsort(l2.real)
    l2_sorted = l2[l2_sorted_indices]
    E_sorted = E[:, l2_sorted_indices]
    E1 = E_sorted[:, 0] / np.linalg.norm(E_sorted[:, 0])
    E2 = E_sorted[:, 1] / np.linalg.norm(E_sorted[:, 1])
    E3 = E_sorted[:, 2] / np.linalg.norm(E_sorted[:, 2])
    eprs, muprs, tauprs = l2_sorted[0].real, l2_sorted[1].real, l2_sorted[2].real
    
    # Neutrinos 
    mv_squared = mv @ mv.conj().T
    m2, V = np.linalg.eig(mv_squared)
    m2_sorted_indices = np.argsort(m2.real)
    m2_sorted = m2[m2_sorted_indices]
    V_sorted = V[:, m2_sorted_indices]
    V1 = V_sorted[:, 0] / np.linalg.norm(V_sorted[:, 0])
    V2 = V_sorted[:, 1] / np.linalg.norm(V_sorted[:, 1])
    V3 = V_sorted[:, 2] / np.linalg.norm(V_sorted[:, 2])
    m1prs, m2prs, m3prs = m2_sorted[0].real, m2_sorted[1].real, m2_sorted[2].real
    
    
    # Computing mixing angles in Vckm and Vpmns
    Vckm_matrix = np.conj(np.array([U1, U2, U3])) @ np.transpose(np.array([D1, D2, D3]))
    Vpmns_matrix = np.conj(np.array([E1, E2, E3])) @ np.transpose(np.array([V1, V2, V3]))
    
    qt12 = np.arctan(np.abs(Vckm_matrix[0, 1]) / np.abs(Vckm_matrix[0, 0]) + 1e-10)
    qt13 = np.arcsin(np.abs(Vckm_matrix[0, 2]))
    qt23 = np.arctan(np.abs(Vckm_matrix[1, 2]) / np.abs(Vckm_matrix[2, 2]) + 1e-10)
    qdcp = np.angle((Vckm_matrix[0, 0] * Vckm_matrix[2, 2] * np.conj(Vckm_matrix[0, 2]) * np.conj(Vckm_matrix[2, 0]) +
                     np.abs(Vckm_matrix[0, 0] * Vckm_matrix[2, 2] * Vckm_matrix[0, 2])**2 / (1 - np.abs(Vckm_matrix[0, 2])**2)))

    vt12 = np.arctan(np.abs(Vpmns_matrix[0, 1]) / np.abs(Vpmns_matrix[0, 0]) + 1e-10)
    vt13 = np.arcsin(np.abs(Vpmns_matrix[0, 2]))
    vt23 = np.arctan(np.abs(Vpmns_matrix[1, 2]) / np.abs(Vpmns_matrix[2, 2]) + 1e-10)
   
    # computing chi^2/n_{obs}
    chiu = ((np.sqrt(uprs) - ubf)**2 / uer**2 + 
            (np.sqrt(cprs) - cbf)**2 / cer**2 + 
            (np.sqrt(tprs) - tbf)**2 / ter**2)
    
    chid = ((np.sqrt(dprs) - dbf)**2 / der**2 + 
            (np.sqrt(sprs) - sbf)**2 / ser**2 + 
            (np.sqrt(bprs) - bbf)**2 / ber**2)
    
    chie = ((np.sqrt(eprs) - ebf)**2 / eer**2 + 
            (np.sqrt(muprs+ 1e-14) - mubf)**2 / muer**2 + 
            (np.sqrt(tauprs) - taubf)**2 / tauer**2)
    
    chiv = ((m2prs - m1prs - delta_ms21)**2 / delta_ms21_er**2 + 
            (m3prs - m1prs - delta_ms31)**2 / delta_ms31_er**2)

    chickm = ((qt12 - q_theta12)**2 / q_theta12_er**2 + 
              (qt13 - q_theta13)**2 / q_theta13_er**2 + 
              (qt23 - q_theta23)**2 / q_theta23_er**2 + 
              (qdcp - q_delta_cp)**2 / q_delta_cp_er**2)

    chipmns = ((np.sin(vt12)**2 - v_theta12)**2 / v_theta12_er**2 + 
               (np.sin(vt13)**2 - v_theta13)**2 / v_theta13_er**2 + 
               (np.sin(vt23)**2 - v_theta23)**2 / v_theta23_er**2).real

    total_chi2 = (chiu + chid + chie + chiv + chickm + chipmns).real / 18

    return total_chi2

def Chi2_M2(s11, s12, s13, s22, s23, s33, d11, d22, d33, a12, a13, a23, alpha, lamb, r1, r4, omega, gamma, m0, neutrino_params):  

    delta_ms21 = neutrino_params['delta_ms21']
    delta_ms31 = neutrino_params['delta_ms31']
    delta_ms21_er = neutrino_params['delta_ms21_er']
    delta_ms31_er = neutrino_params['delta_ms31_er']
    v_theta12 = neutrino_params['v_theta12']
    v_theta13 = neutrino_params['v_theta13']
    v_theta23 = neutrino_params['v_theta23']
    v_theta12_er = neutrino_params['v_theta12_er']
    v_theta13_er = neutrino_params['v_theta13_er']
    v_theta23_er = neutrino_params['v_theta23_er']
    
    I = 1j
    
    S = np.array([[s11, s12, s13],
                  [s12, s22, s23],
                  [s13, s23, s33]])

    D = np.array([[d11, 0, 0],
                  [0, d22, 0],
                  [0, 0, d33]])

    A = np.array([[0, a12, a13],
                  [-a12, 0, a23],
                  [-a13, -a23, 0]])

    # Yukawa matrix in M2
    yu = np.exp(I * alpha)  * S +                              D +        np.exp( I * lamb)  * A
    yd = np.exp(-I * alpha) * S +     r1 * np.exp(I * omega) * D +        np.exp(-I * lamb)  * A
    ye = np.exp(-I * alpha) * S - 3 * r1 * np.exp(I * omega) * D +   r4 * np.exp(-I * gamma)  * A
    yv = np.exp(I * alpha)  * S -                          3 * D +   r4 * np.exp( I * gamma)  * A

    mv = -m0 * np.dot(yv, np.dot(np.diag([1/d11, 1/d22, 1/d33]), yv.T)) 
 
    # Computing eigenvalues and eigenvectors 
    u2, U = np.linalg.eig(yu.astype(complex) @ yu.conj().T)
    d2, D = np.linalg.eig(yd.astype(complex) @ yd.conj().T)
    l2, E = np.linalg.eig(ye.astype(complex) @ ye.conj().T)
    m2, V = np.linalg.eig(mv.astype(complex) @ mv.conj().T)

    # Up-type quarks
    u2_sorted_indices = np.argsort(u2.real)
    u2_sorted = u2[u2_sorted_indices]
    U_sorted = U[:, u2_sorted_indices]
    U1, U2, U3 = U_sorted[:, 0]/np.linalg.norm(U_sorted[:, 0]), U_sorted[:, 1]/np.linalg.norm(U_sorted[:, 1]), U_sorted[:, 2]/np.linalg.norm(U_sorted[:, 2])
    uprs, cprs, tprs = u2_sorted[0].real, u2_sorted[1].real, u2_sorted[2].real

    # Down-type quarks
    d2_sorted_indices = np.argsort(d2.real)
    d2_sorted = d2[d2_sorted_indices]
    D_sorted = D[:, d2_sorted_indices]
    D1, D2, D3 = D_sorted[:, 0]/np.linalg.norm(D_sorted[:, 0]), D_sorted[:, 1]/np.linalg.norm(D_sorted[:, 1]), D_sorted[:, 2]/np.linalg.norm(D_sorted[:, 2])
    dprs, sprs, bprs = d2_sorted[0].real, d2_sorted[1].real, d2_sorted[2].real

    # Charged leptons
    l2_sorted_indices = np.argsort(l2.real)
    l2_sorted = l2[l2_sorted_indices]
    E_sorted = E[:, l2_sorted_indices]
    E1, E2, E3 = E_sorted[:, 0]/np.linalg.norm(E_sorted[:, 0]), E_sorted[:, 1]/np.linalg.norm(E_sorted[:, 1]), E_sorted[:, 2]/np.linalg.norm(E_sorted[:, 2])
    eprs, muprs, tauprs = l2_sorted[0].real, l2_sorted[1].real, l2_sorted[2].real

    # Neutrinos 
    m2_sorted_indices = np.argsort(m2.real)
    m2_sorted = m2[m2_sorted_indices]
    V_sorted = V[:, m2_sorted_indices]
    V1, V2, V3 = V_sorted[:, 0]/np.linalg.norm(V_sorted[:, 0]), V_sorted[:, 1]/np.linalg.norm(V_sorted[:, 1]), V_sorted[:, 2]/np.linalg.norm(V_sorted[:, 2])
    m1prs, m2prs, m3prs = m2_sorted[0].real, m2_sorted[1].real, m2_sorted[2].real

    # Computing mixing angles in Vckm and Vpmns
    Vckm_matrix = np.conj(np.array([U1, U2, U3])) @ np.transpose(np.array([D1, D2, D3]))
    Vpmns_matrix = np.conj(np.array([E1, E2, E3])) @ np.transpose(np.array([V1, V2, V3]))

    qt12 = np.arctan(np.abs(Vckm_matrix[0, 1]) / np.abs(Vckm_matrix[0, 0]))
    qt13 = np.arcsin(np.abs(Vckm_matrix[0, 2]))
    qt23 = np.arctan(np.abs(Vckm_matrix[1, 2]) / np.abs(Vckm_matrix[2, 2]))
    qdcp = np.angle((Vckm_matrix[0, 0] * Vckm_matrix[2, 2] * np.conj(Vckm_matrix[0, 2]) * np.conj(Vckm_matrix[2, 0]) +
                     np.abs(Vckm_matrix[0, 0] * Vckm_matrix[2, 2] * Vckm_matrix[0, 2])**2 / (1 - np.abs(Vckm_matrix[0, 2])**2)))


    vt12 = np.arctan(np.abs(Vpmns_matrix[0, 1]) / np.abs(Vpmns_matrix[0, 0]))
    vt13 = np.arcsin(np.abs(Vpmns_matrix[0, 2]))
    vt23 = np.arctan(np.abs(Vpmns_matrix[1, 2]) / np.abs(Vpmns_matrix[2, 2]))

    # computing chi^2/n_{obs}
    chiu = ((np.sqrt(uprs)  - ubf)**2 / uer**2 + 
            (np.sqrt(cprs)  - cbf)**2 / cer**2 + 
            (np.sqrt(tprs)  - tbf)**2 / ter**2).real
    
    chid = ((np.sqrt(dprs)  - dbf)**2 / der**2 + 
            (np.sqrt(sprs)  - sbf)**2 / ser**2 + 
            (np.sqrt(bprs)  - bbf)**2 / ber**2).real
    
    chie = ((np.sqrt(eprs)  - ebf)**2 / eer**2 + 
            (np.sqrt(muprs)  - mubf)**2 / muer**2 + 
            (np.sqrt(tauprs)  - taubf)**2 / tauer**2).real
    
    chiv = ((m2prs - m1prs - delta_ms21)**2 / delta_ms21_er**2 + 
            (m3prs - m1prs - delta_ms31)**2 / delta_ms31_er**2).real

    chickm = ((qt12 - q_theta12)**2 / q_theta12_er**2 + 
              (qt13 - q_theta13)**2 / q_theta13_er**2 + 
              (qt23 - q_theta23)**2 / q_theta23_er**2 + 
              (qdcp - q_delta_cp)**2 / q_delta_cp_er**2).real

    chipmns = ((np.sin(vt12)**2 - v_theta12)**2 / v_theta12_er**2 + 
               (np.sin(vt13)**2 - v_theta13)**2 / v_theta13_er**2 + 
               (np.sin(vt23)**2 - v_theta23)**2 / v_theta23_er**2).real

    return (chiu + chid + chie + chiv + chickm + chipmns) / 18

def Chi2_M3(s11, s12, s13, s22, s23, s33, d11, d22, d33, a12, a13, a23, r1, r2, r3, r4, r5, m0, neutrino_params):
    
    delta_ms21 = neutrino_params['delta_ms21']
    delta_ms31 = neutrino_params['delta_ms31']
    delta_ms21_er = neutrino_params['delta_ms21_er']
    delta_ms31_er = neutrino_params['delta_ms31_er']
    v_theta12 = neutrino_params['v_theta12']
    v_theta13 = neutrino_params['v_theta13']
    v_theta23 = neutrino_params['v_theta23']
    v_theta12_er = neutrino_params['v_theta12_er']
    v_theta13_er = neutrino_params['v_theta13_er']
    v_theta23_er = neutrino_params['v_theta23_er']
    
    I = 1j
    
    S = np.array([[s11, s12, s13],
                  [s12, s22, s23],
                  [s13, s23, s33]])

    D = np.array([[d11, 0, 0],
                  [0, d22, 0],
                  [0, 0, d33]])

    A = np.array([[0, a12, a13],
                  [-a12, 0, a23],
                  [-a13, -a23, 0]])
    
    # Yukawa matrix in M3
    yu =      S +          D + I *      A
    yd = r2 * S +     r1 * D + I * r3 * A
    ye = r2 * S - 3 * r1 * D + I * r4 * A
    yv =      S - 3 *      D + I * r5 * A

    mv = -m0 * np.dot(yv, np.dot(np.diag([1/d11, 1/d22, 1/d33]), yv.T)) 

    u2, U = np.linalg.eig(yu.astype(complex) @ yu.conj().T)
    d2, D = np.linalg.eig(yd.astype(complex) @ yd.conj().T)
    l2, E = np.linalg.eig(ye.astype(complex) @ ye.conj().T)
    m2, V = np.linalg.eig(mv.astype(complex) @ mv.conj().T)
      
    # Up-type quarks
    u2_sorted_indices = np.argsort(u2.real)
    u2_sorted = u2[u2_sorted_indices]
    U_sorted = U[:, u2_sorted_indices]
    U1, U2, U3 = U_sorted[:, 0]/np.linalg.norm(U_sorted[:, 0]), U_sorted[:, 1]/np.linalg.norm(U_sorted[:, 1]), U_sorted[:, 2]/np.linalg.norm(U_sorted[:, 2])
    uprs, cprs, tprs = u2_sorted[0].real, u2_sorted[1].real, u2_sorted[2].real

    # Down-type quarks
    d2_sorted_indices = np.argsort(d2.real)
    d2_sorted = d2[d2_sorted_indices]
    D_sorted = D[:, d2_sorted_indices]
    D1, D2, D3 = D_sorted[:, 0]/np.linalg.norm(D_sorted[:, 0]), D_sorted[:, 1]/np.linalg.norm(D_sorted[:, 1]), D_sorted[:, 2]/np.linalg.norm(D_sorted[:, 2])
    dprs, sprs, bprs = d2_sorted[0].real, d2_sorted[1].real, d2_sorted[2].real

    # Charged leptons
    l2_sorted_indices = np.argsort(l2.real)
    l2_sorted = l2[l2_sorted_indices]
    E_sorted = E[:, l2_sorted_indices]
    E1, E2, E3 = E_sorted[:, 0]/np.linalg.norm(E_sorted[:, 0]), E_sorted[:, 1]/np.linalg.norm(E_sorted[:, 1]), E_sorted[:, 2]/np.linalg.norm(E_sorted[:, 2])
    eprs, muprs, tauprs = l2_sorted[0].real, l2_sorted[1].real, l2_sorted[2].real

    # Neutrinos 
    m2_sorted_indices = np.argsort(m2.real)
    m2_sorted = m2[m2_sorted_indices]
    V_sorted = V[:, m2_sorted_indices]
    V1, V2, V3 = V_sorted[:, 0]/np.linalg.norm(V_sorted[:, 0]), V_sorted[:, 1]/np.linalg.norm(V_sorted[:, 1]), V_sorted[:, 2]/np.linalg.norm(V_sorted[:, 2])
    m1prs, m2prs, m3prs = m2_sorted[0].real, m2_sorted[1].real, m2_sorted[2].real

    # Computing mixing angles in Vckm and Vpmns
    Vckm_matrix = np.conj(np.array([U1, U2, U3])) @ np.transpose(np.array([D1, D2, D3]))
    Vpmns_matrix = np.conj(np.array([E1, E2, E3])) @ np.transpose(np.array([V1, V2, V3]))

    qt12 = np.arctan(np.abs(Vckm_matrix[0, 1]) / np.abs(Vckm_matrix[0, 0]))
    qt13 = np.arcsin(np.abs(Vckm_matrix[0, 2]))
    qt23 = np.arctan(np.abs(Vckm_matrix[1, 2]) / np.abs(Vckm_matrix[2, 2]))
    qdcp = np.angle((Vckm_matrix[0, 0] * Vckm_matrix[2, 2] * np.conj(Vckm_matrix[0, 2]) * np.conj(Vckm_matrix[2, 0]) +
                     np.abs(Vckm_matrix[0, 0] * Vckm_matrix[2, 2] * Vckm_matrix[0, 2])**2 / (1 - np.abs(Vckm_matrix[0, 2])**2)))

    vt12 = np.arctan(np.abs(Vpmns_matrix[0, 1]) / np.abs(Vpmns_matrix[0, 0]))
    vt13 = np.arcsin(np.abs(Vpmns_matrix[0, 2]))
    vt23 = np.arctan(np.abs(Vpmns_matrix[1, 2]) / np.abs(Vpmns_matrix[2, 2]))

    # computing chi^2/n_{obs}
    chiu = ((np.sqrt(uprs)  - ubf)**2 / uer**2 + 
            (np.sqrt(cprs)  - cbf)**2 / cer**2 + 
            (np.sqrt(tprs)  - tbf)**2 / ter**2).real

    chid = ((np.sqrt(dprs)  - dbf)**2 / der**2 + 
            (np.sqrt(sprs)  - sbf)**2 / ser**2 + 
            (np.sqrt(bprs)  - bbf)**2 / ber**2).real
    
    chie = ((np.sqrt(eprs)  - ebf)**2 / eer**2 + 
            (np.sqrt(muprs)  - mubf)**2 / muer**2 + 
            (np.sqrt(tauprs)  - taubf)**2 / tauer**2).real
    
    chiv = ((m2prs - m1prs - delta_ms21)**2 / delta_ms21_er**2 + 
            (m3prs - m1prs - delta_ms31)**2 / delta_ms31_er**2).real

    chipmns = ((np.sin(vt12)**2 - v_theta12)**2 / v_theta12_er**2 + 
               (np.sin(vt13)**2 - v_theta13)**2 / v_theta13_er**2 + 
               (np.sin(vt23)**2 - v_theta23)**2 / v_theta23_er**2).real
    
    chickm = ((qt12 - q_theta12)**2 / q_theta12_er**2 + 
              (qt13 - q_theta13)**2 / q_theta13_er**2 + 
              (qt23 - q_theta23)**2 / q_theta23_er**2 + 
              (qdcp - q_delta_cp)**2 / q_delta_cp_er**2).real

    return (chiu + chid + chie + chiv + chickm + chipmns) / 18

# ==================== Parameter ranges ====================
def get_parameter_ranges(model_type):
    if model_type == 'M1':
        return {
            'd11': (1e-8, 1e-4),
            'd22': (1e-4, 1e-2),
            'd33': (1e-2, 1),
            's11': (1e-6, 1),
            's12': (1e-6, 1),
            's13': (1e-6, 1),
            's22': (1e-6, 1),
            's23': (1e-6, 1),
            's33': (1e-6, 5),
            's1': (1e-5, 2*np.pi),
            's2': (1e-5, 2*np.pi),
            's3': (1e-5, 2*np.pi),
            's4': (1e-5, 2*np.pi),
            's5': (1e-5, 2*np.pi),
            's6': (1e-5, 2*np.pi),
            'r1': (1e-5, 0.1),
            'r2': (1e-5, 0.1),
            'omega': (1e-5, 2*np.pi),
            'm0': (0.01, 1)
        }
    elif model_type == 'M2':
        return {
            's11': (-1, 1),
            's12': (-1, 1),
            's13': (-1, 1),
            's22': (-1, 1),
            's23': (-1, 1),
            's33': (-1, 1),
            'd11': (5e-12, 1e-5),
            'd22': (1e-5, 2e-3),
            'd33': (2e-3, 1),
            'a12': (-1, 1),
            'a13': (-1, 1),
            'a23': (-1, 1),
            'alpha': (1e-5, 2*np.pi),
            'lamb': (1e-5, 2*np.pi),
            'r1': (1e-5, 3),
            'r4': (1e-5, 3),
            'omega': (1e-5, 2*np.pi),
            'gamma': (1e-5, 2*np.pi),
            'm0': (0.00005, 3)
        }
    elif model_type == 'M3':
        return {
            's11': (-1, 1),      
            's12': (-1, 1), 
            's13': (-1, 1), 
            's22': (-1, 1), 
            's23': (-1, 1),      
            's33': (-1, 1),     
            'd11': (1e-5, 1e-3),  
            'd22': (1e-3, 5e-2),  
            'd33': (5e-2, 1),     
            'a12': (-1, 1), 
            'a13': (-1, 1),    
            'a23': (-1, 1),   
            'r1':  (-1, 1),  
            'r2':  (-1, 1),   
            'r3':  (-1, 1),  
            'r4':  (-1, 1),      
            'r5':  (-1, 1),        
            'm0':  (0.0005, 1)  
        }
    else:
        raise ValueError(f"error input: {model_type}")

# ==================== Model selection ====================
def get_chi2_function(model_type):
    "Return the corresponding chi^2 function based on the model type"
    if model_type == 'M1':
        return Chi2_M1
    elif model_type == 'M2':
        return Chi2_M2
    elif model_type == 'M3':
        return Chi2_M3
    else:
        raise ValueError(f"error input: {model_type}")

# ==================== MPI Initialization ====================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
is_master = (rank == 0)

# ==================== Command-Line Argument Parsing ====================
parser = argparse.ArgumentParser(description="MPI Optimization for GUT Models")
parser.add_argument("--model", type=str, default="M2", choices=["M1", "M2", "M3"])
parser.add_argument("--n-points", type=int, default=100)
parser.add_argument("--n-generations", type=int, default=200)
parser.add_argument("--pop-size", type=int, default=15)
parser.add_argument("--output", type=str, default="mpi_results")
parser.add_argument("--output-dir", type=str, default="./results", help="output")
parser.add_argument("--verbose", action="store_true", help="detailed_output")
parser.add_argument("--octant", type=str, default="1st", choices=["1st", "2nd"])
args = parser.parse_args()

# ==================== Concise Logging Functions ====================
def log_info(message):
    """Print message only from master process, unless verbose mode is enabled"""
    if is_master or args.verbose:
        if is_master:
            print(f"[MASTER] {message}")
        else:
            print(f"[Process {rank}] {message}")

def log_important(message):
    """Display only the most critical information (master process only)"""
    if is_master:
        print(f"*** {message} ***")

# ================ Output Directory Initialization ==================
if is_master:
    # Create output directory (if it doesn't exist)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Display optimization configuration summary
    log_important(f"\n{'='*60}")
    log_important(f"MPI GUT Optimization - Model: {args.model}, Octant: {args.octant}")
    log_important(f"Total Points: {args.n_points}, MPI Processes: {size}")
    log_important(f"DE Settings: {args.n_generations} generations, Population: {args.pop_size}")
    log_important('='*60)

# ==================== Optimizer Class ====================
class DirectOptimizer:
    def __init__(self, model_type, octant='1st'):
        self.model_type = model_type
        self.octant = octant
        self.neutrino_params = setup_neutrino_params(octant)
        self.chi2_func = get_chi2_function(model_type)
        self.param_ranges = get_parameter_ranges(model_type)
        self.param_names = list(self.param_ranges.keys())
        
    def get_bounds(self):
        return [self.param_ranges[name] for name in self.param_names]
    
    def random_point(self):
        params = {}
        for name in self.param_names:
            min_val, max_val = self.param_ranges[name]
            if max_val / min_val > 100 and min_val > 0:
                params[name] = 10**np.random.uniform(np.log10(min_val), np.log10(max_val))
            else:
                params[name] = np.random.uniform(min_val, max_val)
        return params
    
    def params_to_list(self, params_dict):
        return [params_dict[name] for name in self.param_names]
    
    def list_to_params(self, params_list):
        return {name: val for name, val in zip(self.param_names, params_list)}
    
    def evaluate_chi2(self, params):
        if isinstance(params, dict):
            params_list = self.params_to_list(params)
        else:
            params_list = params
        
        try:
            chi2 = self.chi2_func(*params_list, neutrino_params=self.neutrino_params)
            if np.isnan(chi2) or np.isinf(chi2):
                return 1e20
            return float(chi2)
        except Exception:
            return 1e20
    
    def optimize_point(self, params, n_generations, pop_size):
        bounds = self.get_bounds()
        
        try:
            result = differential_evolution(
                self.evaluate_chi2,
                bounds,
                maxiter=n_generations,
                popsize=pop_size,
                recombination=0.7,
                mutation=(0.5, 1.0),
                strategy='best1bin',
                disp=False,
                updating='deferred',
                atol=1e-6,
                tol=0.0001,
                polish=True,
                workers=1
            )
            
            opt_params = self.list_to_params(result.x)
            opt_chi2 = result.fun
            return (opt_chi2, opt_params, True)
            
        except Exception:
            chi2 = self.evaluate_chi2(params)
            return (chi2, params, False)

# ==================== Main Function ====================
def main():
    # Broadcast octant parameter to all processes
    octant = comm.bcast(args.octant if is_master else None, root=0)
    
    # All processes create optimizer with same octant parameter
    optimizer = DirectOptimizer(args.model, octant)
    
    if is_master:
        # Phase 1: Generate initial points
        start_time = time.time()
        
        initial_points = []
        for i in range(args.n_points):
            params = optimizer.random_point()
            chi2 = optimizer.evaluate_chi2(params)
            initial_points.append((chi2, params))
        
        # Display initial statistics
        if initial_points:
            initial_chi2s = [chi2 for chi2, _ in initial_points]
            log_important(f"\nInitial points generated ({args.n_points} points)")
            log_important(f"Initial chi² range: {min(initial_chi2s):.2e} - {max(initial_chi2s):.2e}")
        
        # Phase 2: Distribute tasks
        points_per_proc = args.n_points // size
        remainder = args.n_points % size
        
        # Master process tasks
        my_start = 0
        my_end = points_per_proc + (1 if 0 < remainder else 0)
        my_points = initial_points[my_start:my_end]
        
        # Send tasks to worker processes
        for proc in range(1, size):
            proc_start = my_end
            proc_end = proc_start + points_per_proc + (1 if proc < remainder else 0)
            
            num_points = proc_end - proc_start
            comm.send(num_points, dest=proc, tag=1)
            
            # Send point data
            for i in range(proc_start, proc_end):
                chi2, params = initial_points[i]
                comm.send(chi2, dest=proc, tag=2)
                param_list = optimizer.params_to_list(params)
                comm.send(param_list, dest=proc, tag=3)
            
            my_end = proc_end
        
        log_important(f"Task distribution complete: Master {len(my_points)} points, Total {size} processes")
        
        # Phase 3: Master process optimization
        log_important(f"\nStarting optimization ({args.n_points} points)...")
        opt_start = time.time()
        
        my_results = []
        success_count = 0
        
        for i, (chi2, params) in enumerate(my_points):
            opt_chi2, opt_params, success = optimizer.optimize_point(
                params, args.n_generations, args.pop_size
            )
            
            my_results.append((opt_chi2, opt_params))
            if success:
                success_count += 1
            
            # Display progress at milestones
            if (i + 1) == len(my_points) or ((i + 1) % max(1, len(my_points)//4) == 0):
                elapsed = time.time() - opt_start
                progress = (i + 1) / len(my_points) * 100
                
                if my_results:
                    current_best = min([r[0] for r in my_results])
                    log_important(f"  Progress: {progress:.0f}%, Current best: {current_best:.2e}")
        
        # Phase 4: Collect results
        all_results = my_results.copy()
        
        for proc in range(1, size):
            num_results = comm.recv(source=proc, tag=4)
            
            for _ in range(num_results):
                chi2 = comm.recv(source=proc, tag=5)
                param_list = comm.recv(source=proc, tag=6)
                params = optimizer.list_to_params(param_list)
                all_results.append((chi2, params))
        
        # Collect success counts
        total_success = success_count
        for proc in range(1, size):
            proc_success = comm.recv(source=proc, tag=7)
            total_success += proc_success
        
        # Process results
        all_results.sort(key=lambda x: x[0])
        
        # Build output file path with octant information
        output_file = os.path.join(args.output_dir, f"{args.output}_{args.model}_{octant}.dat")
        
        # Write to file
        with open(output_file, 'w') as f:
            # Write header information
            f.write(f"# Model: {args.model}\n")
            f.write(f"# Octant: {octant}\n")
            f.write(f"# Total points: {len(all_results)}\n")
            f.write(f"# Successful: {total_success}/{args.n_points}\n")
            f.write(f"# Parameters: {', '.join(optimizer.param_names)}\n")
            f.write("#" + "="*80 + "\n")
            
            # Write column headers
            f.write(f"{'chi^2':<20} ")
            for name in optimizer.param_names:
                f.write(f"{name:<15} ")
            f.write("\n")
            
            # Write data (sorted by chi² ascending)
            for chi2, params in all_results:
                f.write(f"{chi2:<20.10e} ")
                param_list = optimizer.params_to_list(params)
                for param in param_list:
                    f.write(f"{param:<15.6e} ")
                f.write("\n")
        
        total_time = time.time() - start_time
        
        # Final statistics
        log_important(f"\n{'='*60}")
        log_important("Optimization complete!")
        
        if all_results:
            # Show only the most important results
            log_important(f"\nBest results:")
            for i in range(min(5, len(all_results))):
                chi2 = all_results[i][0]
                log_important(f"  Rank {i+1}: chi² = {chi2:.6f}")
            
            log_important(f"\nStatistics:")
            log_important(f"  Octant: {octant}")
            log_important(f"  Success rate: {total_success}/{args.n_points} ({total_success/args.n_points*100:.1f}%)")
            log_important(f"  Total time: {total_time:.1f}s")
            log_important(f"  Average per point: {total_time/args.n_points:.1f}s")
            log_important(f"  Output file: {output_file}")
        
        log_important('='*60)
    
    else:
        # Worker process logic
        num_points = comm.recv(source=0, tag=1)
        
        # Receive point data
        my_points = []
        for i in range(num_points):
            chi2 = comm.recv(source=0, tag=2)
            param_list = comm.recv(source=0, tag=3)
            params = optimizer.list_to_params(param_list)
            my_points.append((chi2, params))
        
        # Start optimization (no progress display)
        my_results = []
        success_count = 0
        
        for chi2, params in my_points:
            opt_chi2, opt_params, success = optimizer.optimize_point(
                params, args.n_generations, args.pop_size
            )
            
            my_results.append((opt_chi2, opt_params))
            if success:
                success_count += 1
        
        # Send results
        comm.send(len(my_results), dest=0, tag=4)
        
        for chi2, params in my_results:
            comm.send(chi2, dest=0, tag=5)
            param_list = optimizer.params_to_list(params)
            comm.send(param_list, dest=0, tag=6)
        
        # Send success count
        comm.send(success_count, dest=0, tag=7)
        
        # Only show completion in verbose mode
        if args.verbose:
            log_info(f"Completed {num_points} points (Octant: {octant})")

# ==================== Execution ====================
if __name__ == "__main__":
    main()