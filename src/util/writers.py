

def write_conductivity(frequencies, conductivity, write_obj, is_interacting):
    """
    Write the conductivity to file, given an object to write to.
    :param frequencies: list of frequencies in increaing order
    :param conductivity: list of conductivities corresponding to the
                         frequencies.
    :param write_obj: Object with a .write() method, e.g. open('file', 'w')
    :return:
    """
    interacting_str = 'Interacting' if is_interacting else 'Non-interacting'
    write_obj.write('Optical conductivity (%s)\n' % interacting_str)
    write_obj.write('Omega (s^-1)\t Sigma (e^2/hbar)\n')
    for omega, sigma_omega in zip(frequencies, conductivity):
        write_obj.write('{}\t{}\n'.format(omega, sigma_omega))
