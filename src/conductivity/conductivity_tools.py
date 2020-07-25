from functools import partial

import numpy as np


def get_broadening_function(fnc_name, sigma):
	if fnc_name is 'lorentz':
		fnc = partial(lorentz, sigma=sigma)
	elif fnc_name is 'gauss':
		fnc = partial(gauss, sigma=sigma)
	else:
		raise Exception('Function name not available')
	return fnc

def gauss(w, w0, sigma):
	value = np.exp(-((w-w0)/sigma)**2/2)/np.sqrt(2*np.pi)/sigma
	return value

def lorentz(w, w0, sigma):
	value = sigma/2/((w-w0)**2 + (sigma/2)**2)/np.pi
	return value

def velocity_matrix_element(vb_vector, cb_vector, velocity_matrix):
	matrix_element_vb = np.dot(velocity_matrix, vb_vector)
	matrix_element = np.abs(np.dot(np.conj(cb_vector), matrix_element_vb))**2
	return matrix_element