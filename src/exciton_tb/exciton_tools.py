from itertools import product, chain

import numpy as np
import scipy as sp

recentering_precision = 1e-7
reach_factor = 10

def get_complex_zeros(square_dimension):
    return sp.zeros((square_dimension, square_dimension), dtype=complex)

def cplx_exp_dot(vec1, vec2):
    return np.exp(-1j*np.dot(vec1, vec2))

def conj_dot(vec1, vec2):
    """Returns complex vector inner product <vec1|vec2>"""
    return np.dot(np.conj(vec1.T), vec2)

def hermite_mat_prod(mat_1, mat_2):
    return sp.mat(np.dot(np.conj(mat_1.T), mat_2))

def recentre(m1, m2, nk):
    """Given corner-centred coords, finds midpoint-centred coords."""
    return [m1 - int(m1 > nk // 2)*nk, m2 - int(m2 > nk // 2)*nk]

def recentre_continuous(r, b1, b2):
    threshold = 0.5 + recentering_precision
    x1, x2 = np.dot(b1, r)/2/np.pi, np.dot(b2, r)/2/np.pi
    x1p, x2p = x1 - float(int(x1 > threshold)), x2 - float(int(x2 > threshold))
    return x1p, x2p

def recentre_idx(radius, i, j, nc, a1, a2):
    """Given corner-cetred indices, create spatial vector."""
    rec_idx = recentre((i % nc - j % nc) % nc, (i//nc - j//nc) % nc, nc)
    return radius + rec_idx[0]*a1 + rec_idx[1]*a2

def fix_consistent_gauge(vector):
    """Fix a vector with a consistent gauge depending on its elements"""
    full_sum = vector.ravel().sum()
    quotient_phase = full_sum/np.abs(full_sum)
    fixed_vector = vector/quotient_phase
    return fixed_vector

def get_supercell_positions(a1, a2, nk, cell_wise_centering=True):
    position_list = []
    for m1, m2 in product(range(nk), range(nk)):
        # idx is m1*nk + m2 for reference in future functions
        if cell_wise_centering:
            ms = recentre(m1, m2, nk)
        else:
            ms = m1, m2
        r_position = ms[0]*a1 + ms[1]*a2
        position_list.append(r_position)
    return position_list

def get_cumulative_positions(pattern, norb):

    p_sum = sum(pattern)
    p_len = len(pattern)

    if norb % p_sum != 0:
        raise Exception("Pattern does not fit periodically in orbital set")

    cumul_term = lambda i: (i//p_len)*p_sum + sum(pattern[:(i % p_len) + 1])
    cumulative_positions = [0] + [int(cumul_term(i)) for i in range(norb)]
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

def extract_exciton_dos(excitons,
                        frequencies,
                        broadening_function,
                        sigma,
                        spin_split=False):
    """
    Get the excitonic density of states given a frequency grid and broadening
    function.
    :param excitons: excitonic eigensystem
    :param frequencies: grid of frequencies
    :param broadening_fnc: broadening function (should take only w and w0)
    :param sigma: Broadening used for the broadening function
    :param spin_split: Set to True to loop over spin in the exciton data.
    :return:
    """
    frequency_incr = frequencies[1] - frequencies[0]
    density_of_states = np.zeros(len(frequencies))

    for s0 in range(2):
        if not spin_split and s0 == 1:
            continue
        for idx1, exciton in enumerate(excitons[s0][0]):
            closest_idx = int((exciton - min(frequencies))//frequency_incr)
            reach_idx = reach_factor*int(sigma//frequency_incr)
            lower_idx = max([closest_idx - reach_idx, 0])
            upper_idx = min([closest_idx + reach_idx, len(frequencies) - 1])
            for idx2 in range(lower_idx, upper_idx):
                smearing = broadening_function(exciton, frequencies[idx2])
                density_of_states[idx2] += smearing
    return density_of_states

def convert_eigenvector_convention(eigenvectors, kpt, motif, orb_pattern):
    """
    Change eigenvector block at a k point from convention II to convention I.
    :param eigenvectors: Matrix of eigenvectors
    :param kpt: k point eigenvectors are calculated at
    :param motif: list of motif vectors in order of
    :param orbital_pattern: orbital pattern of number of orbitals per position
    :return:
    """
    nat = len(motif)
    phase_vector = [np.exp(-1j*np.dot(kpt, vector)) for vector in motif]
    full_pattern = [orb_pattern[i % len(orb_pattern)] for i in range(nat)]
    phase_nested_list = [[phase_vector[i]]*full_pattern[i] for i in range(nat)]
    phase_array = np.array(list(chain(*phase_nested_list)))
    eigenvectors_rotate = np.multiply(phase_array.reshape(-1, 1), eigenvectors)
    return eigenvectors_rotate

def convert_all_eigenvectors(eigenvectors,
                             k_grid,
                             motif,
                             orb_pattern,
                             nbasis,
                             spin_split=False):
    """
    Change all eigenvectors from convention II to convention I.
    :param eigenvectors: Matrix of eigenvectors
    :param k_grid: k points that eigenvectors are calculated at
    :param motif: list of motif vectors in order of
    :param orbital_pattern: orbital pattern of number of orbitals per position
    :param nbasis: number of elements in the basis
    :param spin_split: True if system is spin divided.
    :return:
    """
    eigenvectors_rotated = np.zeros(eigenvectors.shape, dtype=complex)
    for idx, kpt in enumerate(k_grid):
        for s in range(2):
            if s == 1 and not spin_split:
                continue
            j1 = idx*nbasis + s*nbasis
            j2 = (idx + 1)*nbasis - int(spin_split)*(1 - s)*nbasis
            eig_block = eigenvectors[j1:j2, :]
            eig_block_rotate = convert_eigenvector_convention(eig_block,
                                                              kpt,
                                                              motif,
                                                              orb_pattern)
            eigenvectors_rotated[j1:j2, :] = eig_block_rotate

    return eigenvectors_rotated


