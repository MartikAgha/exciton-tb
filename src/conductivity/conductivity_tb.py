from itertools import product

import numpy as np

from .conductivity_tools import velocity_matrix_element, \
                                get_broadening_function

e_charge_2_over_epsilon0 = 180.79096


class ConductivityTB:

    polarisation_vectors = {'x': np.array([1, 0]),
                            'y': np.array([0, 1]),
                            'lh': np.array([1j, 1]),
                            'rh': np.array([-1j, 1])}

    def __init__(self, exciton_obj):
        """
        Optical conductivity calculator for tight-binding models
        :param hdf5_input: Input from the tight-binding model in hdf5 format.
        """
        self.exciton_obj = exciton_obj
        self.file_storage = exciton_obj.file_storage
        self.k_grid = exciton_obj.k_grid
        self.a1, self.a2 = exciton_obj.a1, exciton_obj.a2
        self.n_spins = exciton_obj.n_spins
        self.n_orbs = exciton_obj.n_orbs

    def load_velocity_matrix(self, k_idx, s0):
        """
        Load velocity matrix for a specific k_points (and potentially spin).
        :param k_idx: idx of the k-point with respect to self.k_grid
        :param s0: spin index if spin-divided (i.e. n_spins == 2)
        :return: velocity_matrix
        """
        if self.n_spins == 1:
            vel_mat = np.array(self.file_storage['velocity_matrix'][k_idx])
        else:
            vel_mat = np.array(self.file_storage['velocity_matrix'][k_idx][s0])
        return vel_mat

    def convert_str_to_polarisation(self, string):
        """
        Convert polarisation tag to a vector of light polarisation.
        :param string: polarisation tag
        :return: polarisation_vector
        """
        try:
            polarisation_vector = self.polarisation_vectors[string]
        except KeyError:
            polarisation_vector = self.polarisation_vectors['x']
        return polarisation_vector

    def non_interacting_conductivity(self,
                                     sigma=0.04,
                                     freq_range=(0.0, 6.0, 1000),
                                     dielectric_function=False,
                                     broadening='lorentz'):
        """
        Calculate the optical conductivity in the absence of electron-hole
        interactions for a range of frequencies.
        :param sigma: Phenomological broadening given to the conductivity.
        :param freq_range: tuple (freq_start, freq_finish, freq_number) as
                           used for numpy.linspace()
        :param dielectric_function: Set to True to calculate the imaginary
                                    part of the dielectric function.
        :param broadening: type of broadening (e.g. 'lorentz', 'gauss')
        :return:
        """
        system_area = np.abs(np.cross(self.a1, self.a2)[2])*len(self.k_grid)
        broad_fnc = get_broadening_function(broadening, sigma)

        frequency_grid = np.linspace(*freq_range)
        output_grid = np.zeros(len(frequency_grid))

        energy_power = 1 + int(dielectric_function)
        prefactor = e_charge_2_over_epsilon0 if dielectric_function else 1
        prefactor = prefactor/system_area

        n_shift = self.n_spins*self.n_orbs
        for idx_1, kpt in enumerate(self.k_grid):
            for s0 in range(2):
                if self.n_spins == 1 and s0 == 1:
                    continue
                velocity_matrix = self.load_velocity_matrix(idx_1, s0)
                bands = self.exciton_obj.get_number_conduction_valence_bands(
                    idx_1, s0
                )
                v_num, c_num = bands
                eigvals = np.array(
                    self.file_storage['eigensystem']['eigenvalues'][idx_1]
                )

                j1, j2 = n_shift*idx_1, n_shift*(idx_1 + 1)
                eigvecs = np.array(
                    self.file_storage['eigensystem']['eigenvectors'][j1:j2, :]
                )
                for v, c in product(range(v_num), range(c_num)):
                    cb_vector = eigvecs[:, v_num + c]
                    vb_vector = eigvecs[:, v_num]
                    vb_energy, cb_energy = eigvals[v_num + c], eigvals[v_num]
                    matrix_elem = velocity_matrix_element(vb_vector,
                                                          cb_vector,
                                                          velocity_matrix)

                    for idx_2, omega in enumerate(frequency_grid):
                        energy_diff = (cb_energy - vb_energy)**energy_power
                        smearing = broad_fnc(energy_diff, omega)
                        output = prefactor*matrix_elem/energy_diff*smearing
                        output_grid[idx_2] = output

        return output_grid








