from subprocess import Popen
from itertools import product

import numpy as np
import scipy as sp
import scipy.special as spec

def remove(file):
    """Remove file."""
    Popen("rm -rf %s" % file, shell=True)

def get_complex_zeros(square_dimension):
    return sp.zeros((square_dimension, square_dimension), dtype=complex)

def cplx_exp_dot(vec1, vec2):
    return np.exp(-1j*np.dot(vec1, vec2))

def hermite_mat_prod(mat_1, mat_2):
    return sp.mat(np.dot(np.conj(mat_1.T), mat_2))

def tr_keld(radius, radius_0, ep12, alat):
    """Truncated Keldysh potential."""

    main_radius = alat if radius < 1e-5 else radius

    term_arg = (main_radius / radius_0)*ep12
    struve_term = spec.struve(0, term_arg)
    bessel_term = spec.y0(term_arg)

    truncated_keldysh = 22.59/radius_0*(struve_term - bessel_term)

    return truncated_keldysh

def supercell_term(ukc1, ukpc2, ukv1, ukpv2, vkm, nc):
    """BSE matrix element for supercell."""
    matrix_element_vec_left = reduced_tb_vec(ukc1, ukpc2, nc)
    matrix_element_vec_right = np.dot(vkm, reduced_tb_vec(ukpv2, ukv1, nc))

    return np.dot(matrix_element_vec_left, matrix_element_vec_right)

def recentre(m1, m2, nk):
    """Given corner-centred coords, finds midpoint-centred coords."""
    return [m1 - int(m1 > nk // 2)*nk, m2 - int(m2 > nk // 2)*nk]

def recentre_continuous(r, b1, b2):
    x1, x2 = np.dot(b1, r)/2/np.pi, np.dot(b2, r)/2/np.pi
    x1p, x2p = x1 - int(x1 > 0.5), x2 - int(x2 > 0.5)
    return x1p, x2p

def recentre_idx(radius, i, j, nc, a1, a2):
    """Given corner-cetred indices, create spatial vector."""
    rec_idx = recentre((i % nc - j % nc) % nc, (i//nc - j//nc) % nc, nc)
    return radius + rec_idx[0]*a1 + rec_idx[1]*a2

def conj_dot(vec1, vec2):
    """Returns complex vector inner product <vec1|vec2>"""
    return np.dot(np.conj(vec1.T), vec2)

def get_supercell_positions(a1, a2, nk):
    position_list = []
    for m1, m2 in product(range(nk), range(nk)):
        ms = recentre(m1, m2, nk)
        r_position = ms[0]*a1 + ms[1]*a2
        position_list.append(r_position)
    return position_list

def get_cumulative_positions(pattern, norb):

    p_sum = sum(pattern)
    p_len = len(pattern)
    if norb % p_len != 0:
        raise Exception("Pattern does not fit periodically in orbital set")
    cumul_term = lambda i: (i//p_len)*p_sum + sum(pattern[:(i % p_len) + 1])
    cumulative_positions = [0] + [cumul_term(i) for i in range(norb)]
    return cumulative_positions


def reduced_tb_vec(v1, v2, nat, cumul_pos):
    """
    Reduces two TB vectors to a partial-inner-product vector, by collapsing
    only the orbital space.
    :param v1: first vector
    :param v2: second vector
    :param nat: number of atoms in the system
    :param cumul_pos: Cumulative position of each atom in the index list of
                      orbitals.
    :return: reduced vector in the subspace of atoms alone (not orbitals)
    """
    reduced_vec = np.array([
        conj_dot(v1[cumul_pos[i]:cumul_pos[i + 1]],
                 v2[cumul_pos[i]:cumul_pos[i + 1]])
        for i in range(nat)
    ])

    return reduced_vec

def get_band_extrema(eigensystem, energy_limit, cell_size):

    max_energy_1 = eigensystem[0][0][cell_size ** 2 - 1] + energy_limit
    max_energy_2 = eigensystem[1][0][cell_size ** 2 - 1] + energy_limit
    min_energy_1 = eigensystem[0][0][cell_size ** 2] - energy_limit
    min_energy_2 = eigensystem[1][0][cell_size ** 2] - energy_limit

    cb_max_1 = list(eigensystem[0][0] < max_energy_1).index(False) + 1
    cb_max_2 = list(eigensystem[1][0] < max_energy_2).index(False) + 1
    vb_min_1 = list(eigensystem[0][0] >= min_energy_1).index(True)
    vb_min_2 = list(eigensystem[1][0] >= min_energy_2).index(True)

    return cb_max_1, cb_max_2, vb_min_1, vb_min_2
