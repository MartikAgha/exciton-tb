
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


def write_excitons(eigensystem, write_obj, n_spins=1, cutoff_info=None):
    """
    Write the conductivity to file, given an object to write to.
    :param eigensystem: excitonic eigensystem
    :param write_obj: Object with a .write() method, e.g. open('file', 'w')
    :param spin_split: True if exciton output is split by spin
    :return:
    """
    for s0 in range(n_spins):
        write_obj.write('Spin #%d | k_idx/cb_idx/vb_idx\n' % s0)
        write_obj.write('E')
        for item in cutoff_info[s0]:
            k_idx, v_dict = tuple(item)
            transition_range = v_dict['transitions']
            for pair in transition_range:
                head_str = '\t%d/%d/%d' % (k_idx, pair[0], pair[1])
                write_obj.write(head_str)
        write_obj.write('\n')
        for idx, exciton_energy in enumerate(eigensystem[s0][0]):
            exciton_state = eigensystem[s0][1][:, idx].ravel()
            write_obj.write('{}'.format(exciton_energy))
            for element in exciton_state:
                write_obj.write('\t{}'.format(element))
            write_obj.write('\n')
    for cmp, lab in enumerate(['k_x', 'k_y']):
        write_obj.write(lab)
        for item in cutoff_info[0]:
            k_idx, v_dict = tuple(item)
            k_cmp = v_dict['k_point'][cmp]
            write_obj.write('\t%.8f' % k_cmp)
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

