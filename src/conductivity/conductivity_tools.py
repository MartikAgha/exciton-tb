import numpy as np

def gauss(w, w0, sigma):
	value = np.exp(-((w-w0)/sigma)**2/2)/np.sqrt(2*np.pi)/sigma
	return value

def lorentz(w, w0, sigma):
	value = sigma/2/((w-w0)**2 + (sigma/2)**2)/np.pi
	return value