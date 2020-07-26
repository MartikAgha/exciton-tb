
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


def write_excitons(eigensystem, write_obj, spin_split=False):
    """
    Write the conductivity to file, given an object to write to.
    :param eigensystem: excitonic eigensystem
    :param write_obj: Object with a .write() method, e.g. open('file', 'w')
    :param spin_split: True if exciton output is split by spin
    :return:
    """
    for s0 in range(2):
        if not spin_split and s0 == 1:
            continue
        for idx, exciton_energy in enumerate(eigensystem[s0][0]):
            exciton_state = eigensystem[s0][1][:, idx].ravel()
            write_obj.write('{}'.format(exciton_energy))
            for element in exciton_state:
                write_obj.write('\t{}'.format(element))
            write_obj.write('\n')

def write_exciton_dos(frequencies, density_of_states, write_obj):
    """
    Write the conductivity to file, given an object to write to.
    :param frequencies: list of frequencies in increaing order
    :param density_of_states: excitonic eigensystem
    :param write_obj: Object with a .write() method, e.g. open('file', 'w')
    :return:
    """
    write_obj.write('Excitonic Density of States')
    write_obj.write('Omega (s^-1)\t Rho (1/eV)\n')
    for omega, rho in zip(frequencies, density_of_states):
        write_obj.write('{}\t{}\n'.format(omega, rho))

