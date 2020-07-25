from itertools import product

import numpy as np

from .conductivity_tools import velocity_matrix_element, \
                                get_broadening_function

e_charge_2_over_epsilon0 = 180.79096


class ConductivityTB:

    reach_multiplier = 10

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

    def non_interacting_conductivity(self,
                                     sigma=0.04,
                                     freq_range=(0.0, 6.0, 1000),
                                     dielectric_function=False,
                                     broadening='lorentz'):
        """
        Calculate the optical conductivity in the absence of electron-hole
        interactions for a range of frequencies.
        :param sigma: Phenomenological broadening given to the conductivity.
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
        energy_increment = np.abs(frequency_grid[1] - frequency_grid[0])
        num_freqs, min_freq = len(frequency_grid), min(frequency_grid)
        output_grid = np.zeros(num_freqs)

        energy_power = 1 + int(dielectric_function)
        prefactor = e_charge_2_over_epsilon0 if dielectric_function else 1
        prefactor = prefactor/system_area

        n_shift = self.n_spins*self.n_orbs
        for idx1, kpt in enumerate(self.k_grid):
            for s0 in range(2):
                if self.n_spins == 1 and s0 == 1:
                    continue
                velocity_matrix = self.load_velocity_matrix(idx1, s0)
                bands = self.exciton_obj.get_number_conduction_valence_bands(
                    idx1, s0
                )
                v_num, c_num = bands
                eigvals = np.array(
                    self.file_storage['eigensystem']['eigenvalues'][idx1]
                )

                j1, j2 = n_shift*idx1, n_shift*(idx1 + 1)
                eigvecs = np.array(
                    self.file_storage['eigensystem']['eigenvectors'][j1:j2, :]
                )
                for v, c in product(range(v_num), range(c_num)):
                    cb_vector = eigvecs[:, v_num + c]
                    vb_vector = eigvecs[:, v_num]
                    vb_energy, cb_energy = eigvals[v_num + c], eigvals[v_num]
                    energy_diff = (cb_energy - vb_energy)

                    energy_diff_pow = energy_diff**energy_power
                    matrix_elem = velocity_matrix_element(vb_vector,
                                                          cb_vector,
                                                          velocity_matrix)

                    main_term = np.abs(matrix_elem)**2/energy_diff_pow

                    closest_idx = (energy_diff - min_freq)//energy_increment
                    idx_reach = self.reach_multiplier*(sigma//energy_increment)

                    lower_idx = max([closest_idx - idx_reach, 0])
                    upper_idx = min([closest_idx + idx_reach, num_freqs - 1])

                    for idx2 in range(int(lower_idx), int(upper_idx)):
                        smearing = broad_fnc(energy_diff, frequency_grid[idx2])
                        output = prefactor*main_term*smearing
                        output_grid[idx2] += output

        return output_grid

    def interacting_conductivity(self,
                                 sigma=0.04,
                                 freq_range=(0.0, 6.0, 1000),
                                 dielectric_function=False,
                                 broadening='lorentz'):
        """
        Calculate the optical conductivity in the with electron-hole
        interactions for a range of frequencies.
        :param sigma: Phenomenological broadening given to the conductivity.
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

        # Construct vector of velocity matrix elements to dot with the
        # bse eigenvector.
        velocity_elements = []
        for idx_1, kpt in enumerate(self.k_grid):
            for s0 in range(2):
                if self.n_spins == 1 and s0 == 1:
                    continue
                velocity_matrix = self.load_velocity_matrix(idx_1, s0)
                bands = self.exciton_obj.get_number_conduction_valence_bands(
                    idx_1, s0
                )
                v_num, c_num = bands
                j1 = n_shift*idx_1 + s0*self.n_orbs
                j2 = j1 + s0*self.n_orbs
                eigvecs = np.array(
                    self.file_storage['eigensystem']['eigenvectors'][j1:j2, :]
                )
                for c, v in product(range(c_num), range(v_num)):
                    cb_vector = eigvecs[:, v_num + c]
                    vb_vector = eigvecs[:, v_num]
                    matrix_elem = velocity_matrix_element(vb_vector,
                                                          cb_vector,
                                                          velocity_matrix)
                    velocity_elements.append(matrix_elem)

        velocity_elements = np.array(velocity_elements, dtype=complex)
        bse_eigsys = self.exciton_obj.get_bse_eigensystem_direct(solve=True)
        for idx_1 in range(len(bse_eigsys)):
            bse_vector = bse_eigsys[1][:, idx_1]
            bse_value = bse_eigsys[0][idx_1]
            energy_denominator = bse_value**energy_power
            interacting_mat_elem = np.dot(bse_vector, velocity_elements)
            for idx_2, omega in enumerate(frequency_grid):
                smearing = broad_fnc(bse_value, omega)
                main_term = np.abs(interacting_mat_elem)**2/energy_denominator
                output = prefactor*main_term*smearing
                output_grid[idx_2] += output

        return output_grid

