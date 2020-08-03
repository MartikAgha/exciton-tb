from functools import partial

import numpy as np


def get_broadening_function(fnc_name, sigma):
    if fnc_name == 'lorentz':
        fnc = partial(lorentz, sigma=sigma)
    elif fnc_name == 'gauss':
        fnc = partial(gauss, sigma=sigma)
    else:
        raise Exception('Function name not available')
    return fnc


def gauss(w, w0, sigma):
    value = np.exp(-((w - w0)/sigma)**2/2)/np.sqrt(2 * np.pi)/sigma
    return value


def lorentz(w, w0, sigma):
    value = sigma/2/((w - w0)**2 + (sigma/2)**2)/np.pi
    return value


def velocity_matrix_element(vb_vector, cb_vector, velocity_matrix):
    matrix_element_vb = np.dot(velocity_matrix, vb_vector)
    matrix_element = np.dot(np.conj(cb_vector), matrix_element_vb)
    return matrix_element

def residual_term(vec_c, vec_v, val_c, val_v, motif_polarised):
    prefactor = val_v - val_c
    vec_v_times_motif = np.multiply(motif_polarised, vec_v)
    position_matrix_element = np.dot(np.conj(vec_c), vec_v_times_motif)
    residual = prefactor*position_matrix_element
    return residual
