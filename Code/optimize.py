# ======== The code for numerical scanning in arXiv:2511.16196  ========
# =====  Authors: Zi-Qiang Chen, Gao-Xiang Fang and Ye-Ling Zhou   =====

import numpy as np
import random
from mpi4py import MPI
import argparse
import time
import os
from itertools import product

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
                  [0, d22, 0],
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
            'd11':  (1e-7, 1e-5),
            'd22':  (1e-5, 1e-3),
            'd33':  (1e-3, 0.01),
            's11': (1e-7, 1e-1),
            's12': (1e-7, 1e-1),
            's13': (1e-7, 1e-1),
            's22': (1e-7, 1e-1),
            's23': (1e-6, 1e-1),
            's33': (1e-6, 1e-1),
            's1': (0, 2*np.pi),
            's2': (0, 2*np.pi),
            's3': (0, 2*np.pi),
            's4': (0, 2*np.pi),
            's5': (0, 2*np.pi),
            's6': (0, 2*np.pi),
            'r1': (10, 300),
            'r2': (1e-3, 1),
            'omega': (0, 2*np.pi),
            'm0': (0.0005, 1)
        }
    elif model_type == 'M2':
        return {
            's11': (-0.02, 0.02),
            's12': (-0.1, 0.1),
            's13': (-1, 1),
            's22': (-1, 1),
            's23': (-5, 5),
            's33': (-6, 6),
            'd11': (1e-8, 1e-4),
            'd22': (1, 2),
            'd33': (420, 440),
            'a12': (-0.1, 0.1),
            'a13': (-1, 1),
            'a23': (-5, 5),
            'alpha': (0, 2*np.pi),
            'lamb': (0, 2*np.pi),
            'r1': (1e-5, 0.01),
            'r4': (0.1, 3),
            'omega': (0, 2*np.pi),
            'gamma': (0, 2*np.pi),
            'm0': (0.0005, 1)
        }
    elif model_type == 'M3':
        return {
            's11': (-0.01, 0.01),      
            's12': (-0.01, 0.01), 
            's13': (-0.01, 0.01), 
            's22': (-0.01, 0.01), 
            's23': (-0.01, 0.01),      
            's33': (-1, 1),     
            'd11': (1e-5, 1e-3),  
            'd22': (5e-3, 1e-1),  
            'd33': (1e-2, 0.5),     
            'a12': (-0.01, 0.01), 
            'a13': (-0.1, 0.1),    
            'a23': (-0.1, 0.1),   
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

# ==================== MPI stage 1 optimization ====================
def optimize_for_initial_point_mpi(model_type, learning_rate, max_iterations, parameter_ranges, min_lr, max_lr, neutrino_params, seed_offset=0):
    "optimize with MPI"
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # get chi^2 values
    chi2_func = get_chi2_function(model_type)
    
    # introduce parameter ranges
    if isinstance(parameter_ranges, dict):
        param_ranges_list = list(parameter_ranges.values())
    else:
        param_ranges_list = parameter_ranges
        
    num_params = len(param_ranges_list)
    
    # seeding
    random.seed(rank + seed_offset)
    np.random.seed(rank + seed_offset)
    
    adaptive_lr = learning_rate
    no_improvement_count = 0
    
    report_interval = max(1, max_iterations // 10)
    patience = max(10, max_iterations // 20)
    
    # random initial ranges
    current_params = [float(random.uniform(param_ranges_list[i][0], param_ranges_list[i][1])) 
                     for i in range(num_params)]
    
    try:
        current_loss = float(chi2_func(*current_params, neutrino_params=neutrino_params))
    except Exception as e:
        current_loss = float('inf')
    
    best_loss = current_loss
    best_params = current_params.copy()
    
    for current_iteration in range(1, max_iterations + 1):
        try:
            # Generate random perturbations
            rand_params = [float(random.uniform(-adaptive_lr, adaptive_lr)) for _ in range(num_params)]
            
            # Generate candidate parameters
            candidate_params = []
            for i in range(num_params):
                new_val = float(current_params[i] + rand_params[i])
                new_val = max(param_ranges_list[i][0], min(param_ranges_list[i][1], new_val))
                candidate_params.append(new_val)
            
            candidate_loss = float(chi2_func(*candidate_params, neutrino_params=neutrino_params))
            
            if candidate_loss < current_loss:
                current_params = candidate_params.copy()
                current_loss = candidate_loss
                no_improvement_count = 0
                
                if candidate_loss < best_loss:
                    best_loss = candidate_loss
                    best_params = candidate_params.copy()
                    adaptive_lr = min(max_lr, adaptive_lr * 1.05)
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    adaptive_lr = max(min_lr, adaptive_lr * 0.5)
                    no_improvement_count = 0
            
            if report_interval > 0 and current_iteration % report_interval == 0:
                adaptive_lr = max(min_lr, min(max_lr, adaptive_lr))
                
        except Exception as e:
            continue
    
    return (float(best_loss), [float(x) for x in best_params])

# ==================== MPI stage 2 optimization ====================
def parallel_table_search_momentum_mpi(model_type, initial_point, total_steps=10000, batch_size=8, 
                                     parameter_ranges=None, neutrino_params=None):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    chi2_func = get_chi2_function(model_type)
    
    if isinstance(parameter_ranges, dict):
        param_ranges_list = list(parameter_ranges.values())
    else:
        param_ranges_list = parameter_ranges
    
    # Compute the stage 2 initial points 
    if len(initial_point) == len(param_ranges_list) + 1:
        initial_params = np.array(initial_point[1:])
        initial_chi2 = initial_point[0]
    else:
        initial_params = np.array(initial_point)
        initial_chi2 = chi2_func(*initial_point, neutrino_params=neutrino_params)
    
    dim = len(initial_params)
    best_point = initial_params.copy()
    best_chi2 = initial_chi2
    
    momentum = 0.0
    step_size = 0.00003
    last_improvement = 0
    
    # Seeding
    np.random.seed(rank + int(time.time() * 1000) % 1000000)
    
    for step in range(1, total_steps + 1):
        improvement = 0
        
        batch_points = []
        for _ in range(batch_size):
            random_perturbation = np.random.uniform(-1, 1, dim)
            perturbation = (step_size + momentum) * random_perturbation
            new_point = best_point * (1 + perturbation)
            batch_points.append(new_point)
            
        # Calculate the batch chi-square value
        batch_chi2s = []
        for point in batch_points:
            batch_chi2s.append(chi2_func(*point, neutrino_params=neutrino_params))
        
        for i, chi2 in enumerate(batch_chi2s):
            if chi2 < best_chi2:
                improvement = best_chi2 - chi2
                best_point = batch_points[i].copy()
                best_chi2 = chi2
                last_improvement = step
        
        # momentum adjustment
        if improvement > 0:
            momentum = min(0.000003, momentum + 0.000003)
        else:
            momentum = max(0, momentum - 0.000006)
        
        # Adjust the step size when there is no improvement for a long time
        if step - last_improvement > 1000:
            step_size = min(0.000001, step_size * 1.1)
            last_improvement = step
    
    return best_chi2, best_point.tolist()

# ==================== Main optimization function of MPI ====================
def mpi_optimize_stage1(model_type, total_initial_points, iterations_per_point, learning_rate, 
                       parameter_ranges, neutrino_params, keep_points):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Timing
    stage1_start_time = time.time()
    
    if rank == 0:
        print(f"MPI Stage 1 Optimization: Total initial points={total_initial_points}, Number of processes={size}")
    
    # Calculate the tasks of each process
    points_per_process = total_initial_points // size
    remainder = total_initial_points % size
    
    if rank < remainder:
        my_points = points_per_process + 1
        start_idx = rank * (points_per_process + 1)
    else:
        my_points = points_per_process
        start_idx = remainder * (points_per_process + 1) + (rank - remainder) * points_per_process
    
    my_results = []
    
    # Start timing
    process_start_time = time.time()
    
    for i in range(my_points):
        point_id = start_idx + i
        
        try:
            best_loss, best_params = optimize_for_initial_point_mpi(
                model_type=model_type,
                learning_rate=learning_rate,
                max_iterations=iterations_per_point,
                parameter_ranges=parameter_ranges,
                min_lr=0.0001,
                max_lr=learning_rate,
                neutrino_params=neutrino_params,
                seed_offset=point_id * 1000
            )
            
            # Create simple serializable results
            result_row = [float(best_loss)] + [float(x) for x in best_params]
            my_results.append(result_row)
            
        except Exception as e:
            continue
    
    # Compute the total time for stage 1
    process_time = time.time() - process_start_time
    
    try:
        all_results = comm.gather(my_results, root=0)
        all_times = comm.gather(process_time, root=0)
    except Exception as e:
        if rank == 0:
            print(f"MPI collection failed: {e}")
        return None
    stage1_total_time = time.time() - stage1_start_time
    
    if rank == 0:
        # Merge all the results
        combined_results = []
        for process_results in all_results:
            if process_results is not None:
                combined_results.extend(process_results)
        
        # Sort and select the best
        if combined_results:
            combined_results.sort(key=lambda x: x[0])
            final_results = combined_results[:keep_points]
            
            print(f"Stage 1 completed: Kept {len(final_results)} best points from {len(combined_results)} points")
            print(f"Best Chi2: {final_results[0][0]:.6f}")
            print(f"Stage 1 total time: {stage1_total_time:.2f}s")
            
            # Display process time statistics
            if all_times:
                avg_time = sum(all_times) / len(all_times)
                max_time = max(all_times)
                min_time = min(all_times)
                print(f"  Average: {avg_time:.2f}s, Longest: {max_time:.2f}s, Shortest: {min_time:.2f}s")
                print(f"  Load balancing efficiency: {avg_time/max_time*100:.1f}%")
            
            return final_results
        else:
            print("Error: No valid results")
            return None
    else:
        return None
        
def mpi_optimize_stage2(model_type, stage1_results, total_steps, batch_size, 
                       parameter_ranges, neutrino_params):
    """MPI Stage 2 optimization main function"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"MPI Stage 2 Optimization: Input points={len(stage1_results)}, Processes={size}")
    
    # Distribute tasks to each process
    if rank == 0:
        points_per_process = len(stage1_results) // size
        remainder = len(stage1_results) % size
        
        tasks = []
        for i in range(size):
            if i < remainder:
                start = i * (points_per_process + 1)
                end = start + points_per_process + 1
            else:
                start = remainder * (points_per_process + 1) + (i - remainder) * points_per_process
                end = start + points_per_process
            tasks.append(stage1_results[start:end])
    else:
        tasks = None
    
    # Distribute tasks
    my_tasks = comm.scatter(tasks, root=0)
    
    # Optimize own tasks on this process
    my_results = []
    start_time = time.time()
    
    for i, task in enumerate(my_tasks):
        best_chi2, best_params = parallel_table_search_momentum_mpi(
            model_type=model_type,
            initial_point=task,
            total_steps=total_steps,
            batch_size=batch_size,
            parameter_ranges=parameter_ranges,
            neutrino_params=neutrino_params
        )
        
        my_results.append([best_chi2] + best_params)
        
    #    if (i + 1) % max(1, len(my_tasks) // 5) == 0:
    #        elapsed = time.time() - start_time
    #        print(f"Process {rank}: Completed {i+1}/{len(my_tasks)} Stage 2 optimization points")
    
    # Collect all results
    all_results = comm.gather(my_results, root=0)
    
    if rank == 0:
        # Merge all results
        combined_results = []
        for process_results in all_results:
            combined_results.extend(process_results)
        
        # Sort results
        combined_results.sort(key=lambda x: x[0])
        
        stage2_time = time.time() - start_time
        print(f"Stage 2 completed: {stage2_time:.2f}s")
        print(f"Final best Chi2: {combined_results[0][0]:.6f}")
        
        return combined_results
    else:
        return None

# ==================== MPI main function ====================
def main_mpi():
    """MPI main function"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Only rank 0 parses arguments
    if rank == 0:
        parser = argparse.ArgumentParser(description='MPI multi-node two-stage optimization program')
        
        # General parameters
        parser.add_argument('--model', type=str, choices=['M1', 'M2', 'M3'], required=True,
                          help='Model type to optimize')
        parser.add_argument('--output-dir', type=str, default='./mpi_results', help='Output directory')
        parser.add_argument('--run-mode', type=str, choices=['stage1', 'stage2', 'both'], default='both', 
                          help='Run mode')
        parser.add_argument('--neutrino-version', type=str, choices=['1st', '2nd'], default='1st',
                          help='Neutrino parameter version')
        
        # Stage 1 parameters - using correct short names
        parser.add_argument('--total-initial-points', type=int, default=1024, 
                          help='Total initial points for Stage 1')
        parser.add_argument('--iterations-per-point', type=int, default=50000, 
                          help='Iterations per point in Stage 1')
        parser.add_argument('--learning-rate', type=float, default=0.1, 
                          help='Learning rate for Stage 1')
        parser.add_argument('--keep-points', type=int, default=256,
                          help='Number of best points to keep from Stage 1')
        
        # Stage 2 parameters - using correct short names
        parser.add_argument('--total-steps', type=int, default=2000, 
                          help='Total steps for Stage 2')
        parser.add_argument('--batch-size', type=int, default=32, 
                          help='Batch size for Stage 2')
        
        args = parser.parse_args()
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        print("MPI multi-node two-stage optimization program started")
        print(f"Model type: {args.model}")
        print(f"Neutrino version: {args.neutrino_version}")
        print(f"Number of processes: {size}")
        print(f"Run mode: {args.run_mode}")
        print(f"Output directory: {args.output_dir}")
        
        if args.run_mode in ['stage1', 'both']:
            print(f"Stage 1: Total points={args.total_initial_points}, Iterations per point={args.iterations_per_point}")
            print(f"        Learning rate={args.learning_rate}, Keep points={args.keep_points}")
        
        if args.run_mode in ['stage2', 'both']:
            print(f"Stage 2: Total steps={args.total_steps}, Batch size={args.batch_size}")
        
    else:
        args = None
    
    # Broadcast parameters
    args = comm.bcast(args, root=0)
    
    # Set neutrino parameters and parameter ranges
    neutrino_params = setup_neutrino_params(args.neutrino_version)
    parameter_ranges = get_parameter_ranges(args.model)
    
    # Stage 1 optimization
    stage1_results = None
    if args.run_mode in ['stage1', 'both']:
        if rank == 0:
            print(f"\n=== Starting MPI Stage 1 Optimization ===")
        
        stage1_results = mpi_optimize_stage1(
            model_type=args.model,
            total_initial_points=args.total_initial_points,  
            iterations_per_point=args.iterations_per_point,  
            learning_rate=args.learning_rate, 
            parameter_ranges=parameter_ranges,
            neutrino_params=neutrino_params,
            keep_points=args.keep_points 
        )
        
        # Save Stage 1 results
        if rank == 0 and stage1_results is not None:
            stage1_file = os.path.join(args.output_dir, f'{args.model}_{args.neutrino_version}_stage1.dat')
            with open(stage1_file, 'w') as f:
                f.write(f"# MPI Stage 1 optimization results - Model: {args.model}, Neutrino version: {args.neutrino_version}\n")
                f.write(f"# Total search points: {args.total_initial_points}, Kept points: {len(stage1_results)}\n")
                for row in stage1_results:
                    f.write(" ".join(f"{x:.12e}" for x in row) + "\n")
            print(f"Stage 1 results saved to: {stage1_file}")
    
    # Stage 2 optimization
    if args.run_mode in ['stage2', 'both']:
        if rank == 0:
            print(f"\n=== Starting MPI Stage 2 Optimization ===")
            
            # If loading Stage 1 results from file
            if stage1_results is None and args.run_mode == 'stage2':
                stage1_file = os.path.join(args.output_dir, f'{args.model}_{args.neutrino_version}_stage1.dat')
                if os.path.exists(stage1_file):
                    stage1_results = []
                    with open(stage1_file, 'r') as f:
                        for line in f:
                            if line.startswith('#'):
                                continue
                            values = list(map(float, line.strip().split()))
                            stage1_results.append(values)
                    print(f"Loaded {len(stage1_results)} Stage 1 results from file")
                else:
                    print(f"Error: Stage 1 results file not found: {stage1_file}")
                    return
        
        # Broadcast Stage 1 results to all processes
        stage1_results = comm.bcast(stage1_results, root=0)
        
        final_results = mpi_optimize_stage2(
            model_type=args.model,
            stage1_results=stage1_results,
            total_steps=args.total_steps,
            batch_size=args.batch_size,
            parameter_ranges=parameter_ranges,
            neutrino_params=neutrino_params
        )
        
        # Save final results
        if rank == 0 and final_results is not None:
            final_file = os.path.join(args.output_dir, f'{args.model}_{args.neutrino_version}.dat')
            with open(final_file, 'w') as f:
                f.write(f"# MPI final optimization results - Model: {args.model}, Neutrino version: {args.neutrino_version}\n")
                f.write(f"# Input points: {len(stage1_results)}, Output points: {len(final_results)}\n")
                for row in final_results:
                    f.write(" ".join(f"{x:.12e}" for x in row) + "\n")
            
            print(f"\n=== MPI optimization completed ===")
            print(f"Final results saved to: {final_file}")
            print(f"Best Chi2: {final_results[0][0]:.6f}")
            
            # Display top 5 best results
            print(f"\nTop 5 best results:")
            for i in range(min(5, len(final_results))):
                print(f"  {i+1}. Chi2 = {final_results[i][0]:.6f}")
                
if __name__ == "__main__":
    main_mpi()