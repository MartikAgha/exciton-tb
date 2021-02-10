#!/usr/bin/python3
import os
from itertools import product

import numpy as np
import h5py as hp
import numpy.linalg as napla

from .exciton_tools import get_complex_zeros, cplx_exp_dot, \
                           get_cumulative_positions, reduced_tb_vec, \
                           recentre_continuous, get_supercell_positions, \
                           fix_consistent_gauge, convert_all_eigenvectors, \
                           orthogonalize_eigenvecs
from .exciton_interactions import interaction_potential


class ExcitonTB:
    
    spin_str = 's%d'
    one_point_str = 'k(%d)'
    two_point_str = 'k(%d,%d)'
    four_point_str = 'k(%d,%d,%d,%d)'

    hermitian_msg = "{} != {} , Not hermitian"
    hermitian_rounding_dp = 8
    round_dp = 10

    # If args == None, then these will be used.
    default_args = {'radius_0': 33.875,
                    'substrate_dielectric': 3.8,
                    'gamma': 0.1}

    # Location matrix elements wil be stored.
    matrix_element_dir = '../matrix_element_hdf5'

    def __init__(self, hdf5_input, potential_name='keldysh', args=None,
                 cell_wise_centering=True):
        """
        Exciton calculator for tight-binding models.
        :param hdf5_input: name of hdf5 file that contains crystal information
                           and eigensystem
        :param potential_name: type of potential used (only keldysh for now)
        :param args: potential parameters: Dictionary containing interaction
                                           parameters
                     {'radius_0': Inherent screening value for keldysh int.,
                      'substrate_dielectric': dielectric const. of substrate,
                      'gamma': Potential decay value for Yukawa potential}
        """
        self.potential_name = potential_name
        self.interaction_args = self.default_args if args is None else args

        # Remains open for the calculation, use self.terminate_storage_usage
        # when done using this.
        self.file_storage = hp.File(hdf5_input, 'r')
        # Extract geometric information
        self.alat = float(np.array(self.file_storage['crystal']['alat']))
        self.a1 = np.array(self.file_storage['crystal']['avecs']['a1']).ravel()
        self.a2 = np.array(self.file_storage['crystal']['avecs']['a2']).ravel()
        self.b1 = np.array(self.file_storage['crystal']['gvecs']['b1']).ravel()
        self.b2 = np.array(self.file_storage['crystal']['gvecs']['b2']).ravel()
        self.motif_vectors = np.array(self.file_storage['crystal']['motif'])

        self.n_atoms = int(np.array(self.file_storage['crystal']['n_atoms']))
        self.n_orbs = int(np.array(self.file_storage['crystal']['n_orbs']))
        self.n_k = int(np.array(self.file_storage['eigensystem']['n_k']))
        self.n_val = int(np.array(self.file_storage['eigensystem']['n_val']))
        self.n_con = int(np.array(self.file_storage['eigensystem']['n_con']))

        self.n_spins = int(np.array(
            self.file_storage['eigensystem']['n_spins']
        ))
        self.convention = int(np.array(
            self.file_storage['eigensystem']['convention']
        ))

        if self.n_spins not in [1, 2]:
            raise ValueError("eigensystem/n_spins must be either 1 or 2.")

        self.is_complex = bool(np.array(
            self.file_storage['eigensystem']['is_complex']
        ))

        # Acrue real-space and reciprocal-space grid.
        self.k_grid = np.array(self.file_storage['eigensystem']['k_grid'])
        self.cell_wise_centering = cell_wise_centering
        self.r_grid = get_supercell_positions(self.a1,
                                              self.a2,
                                              self.n_k,
                                              cell_wise_centering)

        # Value of the position to use ear R-->0 in the interaction potential.
        self.trunc_alat = float(np.array(
            self.file_storage['crystal']['trunc_alat']
        ))
        self.orb_pattern = list(map(int, list(np.array(
            self.file_storage['crystal']['orb_pattern']).ravel()
        )))
        self.cumulative_positions = get_cumulative_positions(self.orb_pattern,
                                                             self.n_orbs)

        self.selective_mode = bool(np.array(self.file_storage['selective']))
        if self.selective_mode:
            num_states = np.array(
                self.file_storage['band_edges']['num_states']
            )
            try:
                self.num_states = int(num_states)
            except TypeError:
                self.num_states = list(num_states)

        if not os.path.exists(self.matrix_element_dir):
            os.mkdir(self.matrix_element_dir)

        self.element_storage_name = None

    def create_matrix_element_hdf5(self, storage_name, energy_cutoff=None):
        """
        Create matrix elements for the direct coulomb interaction
        :param storage_name: name for the hdf5 storage for matrix elements.
        :return:
        """
        self.element_storage_name = os.path.join(self.matrix_element_dir,
                                                 storage_name)
        # Acquire atomic structure parameters.
        nat = self.n_atoms
        norb = self.n_orbs
        nk, nk2 = self.n_k, self.n_k**2

        # Containers for outputted data
        pos_list = self.r_grid

        # Place holder function to get cutoff_band_min_maxes
        cutoff_band_min_maxes = self.get_cutoff_band_min_maxes(energy_cutoff)

        with hp.File(self.element_storage_name, 'w') as f:
            eigvecs = self.get_preprocessed_eigenvectors()
            eh_int = self.get_eh_int(nat, nk, pos_list)
            f['VkM'] = eh_int
            f['use_energy_cutoff'] = bool(energy_cutoff is not None)
            f['energy_cutoff'] = 0 if energy_cutoff is None else energy_cutoff
            cutoff_bands = f.create_group('cutoff_bands')
            nk_shift = nk if nk == 1 else 2*nk

            state_shift = self.n_spins*norb

            for s0 in range(self.n_spins):
                # Two separate sets for up and down spin for non-exchange
                # Assuming a Gamma-point calculation for now in supercell
                spins = f.create_group(self.spin_str % s0)
                spins_cutoff = cutoff_bands.create_group(self.spin_str % s0)

                for m1, m2 in product(range(nk), range(nk)):
                    ki_m = m1*nk + m2
                    # For each k point, record the new c_num and v_max
                    k_minmax = spins_cutoff.create_group('(%d,%d)' % (m1, m2))
                    k_minmax['v_min'] = cutoff_band_min_maxes[s0][ki_m][0]
                    k_minmax['c_max'] = cutoff_band_min_maxes[s0][ki_m][1]

                    for l1, l2 in product(range(nk), range(nk)):
                        ki_l = l1*nk + l2

                        # Indices to navigate through k,k' data
                        ml1 = m1 - l1 + nk if nk > 1 else 0
                        ml2 = m2 - l2 + nk if nk > 1 else 0
                        # ml1 = (m1 - l1) % nk
                        # ml2 = (m2 - l2) % nk

                        bandnum_m = self.get_number_conduction_valence_bands(
                            ki_m, s0
                        )
                        v_num_m, c_num_m = bandnum_m

                        bandnum_l = self.get_number_conduction_valence_bands(
                            ki_l, s0
                        )
                        v_num_l, c_num_l = bandnum_l

                        if energy_cutoff is not None:
                            cutoff_minmax_m = cutoff_band_min_maxes[s0][ki_m]
                            cutoff_minmax_l = cutoff_band_min_maxes[s0][ki_l]
                            v_min_m = int(cutoff_minmax_m[0])
                            c_num_m = int(cutoff_minmax_m[1]) + 1
                            v_min_l = int(cutoff_minmax_l[0])
                            c_num_l = int(cutoff_minmax_l[1]) + 1
                            v_num_m = v_num_m - v_min_m
                            v_num_l = v_num_l - v_min_l
                        else:
                            v_min_m, v_min_l = 0, 0

                        valence = np.zeros((nat, v_num_m*v_num_l),
                                           dtype='complex')
                        conduction = np.zeros((nat, c_num_m*c_num_l),
                                              dtype='complex')

                        m_shf, l_shf = ki_m*state_shift, ki_l*state_shift
                        id_string = self.four_point_str % (m1, m2, l1, l2)
                        kpts = spins.create_group(id_string)

                        e_m, e_l = m_shf + s0*norb, l_shf + s0*norb

                        vb_iter = product(range(v_num_m), range(v_num_l))
                        for v1, v2 in vb_iter:
                            # for v1, k' and v2, k find matrix element
                            # This is reversed compared to conduction band
                            vm_i = v2*v_num_m + v1
                            valence[:, vm_i] = reduced_tb_vec(
                                eigvecs[e_l:e_l + norb, v_min_l + v2],
                                eigvecs[e_m:e_m + norb, v_min_m + v1],
                                nat,
                                self.cumulative_positions
                            )

                        cb_iter = product(range(c_num_m), range(c_num_l))
                        for c1, c2 in cb_iter:
                            # for c1, k and c2, k' find matrix element
                            cm_i = c1*c_num_l + c2
                            conduction[:, cm_i] = reduced_tb_vec(
                                eigvecs[e_m:e_m + norb, v_min_m + v_num_m + c1],
                                eigvecs[e_l:e_l + norb, v_min_l + v_num_l + c2],
                                nat,
                                self.cumulative_positions
                            )

                        elem_shape = (c_num_m*c_num_l, v_num_m*v_num_l)
                        elems = kpts.create_dataset(name='mat_elems',
                                                    shape=elem_shape,
                                                    dtype='complex')

                        elems[:] = np.dot(np.array(conduction).T,
                                          np.dot(eh_int[ml1*nk_shift + ml2],
                                                 np.array(valence)))

                        del conduction, valence

    def get_bse_eigensystem_direct(self,
                                   matrix_element_storage=None,
                                   solve=True):
        """
        Construct the Hamiltonian (and solve it if necessary) for the
        Bethe-Salpeter Equation for the direct photonic transitions (Q=0).
        :param matrix_element_storage: HDF5 Storage for the matrix elements.
        :param solve: Set to True to find the eigenvalues, otherwise will
                      output the matrix.
        :return:
        """
        if matrix_element_storage is None:
            element_storage = self.element_storage_name
        else:
            element_storage = matrix_element_storage

        g = hp.File(element_storage, 'r')
        energy_cutoff_bool = bool(np.array(g['use_energy_cutoff']))
        nk, nk2 = self.n_k, self.n_k**2
        n_val, n_con = self.n_val, self.n_con
        selective = self.selective_mode

        mat_dim, blocks = self.get_matrix_dim_and_block_starts(g)
        block_skip = n_con*n_val
        spin_shift = n_con + n_val
        bse_mat = [get_complex_zeros(mat_dim[i]) for i in range(self.n_spins)]
        q_zero = np.array([0, 0])

        for m1, m2 in product(range(nk), range(nk)):
            k_i = nk*m1 + m2
            # Diagonal elements: Use cb/vb energy differences
            energy_k, energy_kq = self.get_diag_eigvals(k_idx=k_i,
                                                        kq_idx=k_i,
                                                        q_crys=q_zero,
                                                        direct=True)
            for s0 in range(self.n_spins):
                k_skip = k_i*block_skip

                if selective or energy_cutoff_bool:
                    k_skip = blocks[s0][k_i]

                vnum1, cnum1 = self.get_number_conduction_valence_bands(
                    k_i, s0
                )

                # This should be defined before we change v_num and c_num to
                # account for energy cutoff.
                spin_skip = s0*(vnum1 + cnum1) if selective else s0*spin_shift

                if energy_cutoff_bool:
                    k_str = '(%d,%d)' % (m1, m2)
                    s_str = self.spin_str % s0
                    v_min = int(np.array(
                        g['cutoff_bands'][s_str][k_str]['v_min']
                    ))
                    cnum1 = int(np.array(
                        g['cutoff_bands'][s_str][k_str]['c_max']
                    )) + 1
                    vnum1 = vnum1 - v_min
                else:
                    v_min = 0

                for c, v in product(range(cnum1), range(vnum1)):
                    mat_idx = k_skip + c*vnum1 + v
                    final_energy = energy_kq[v_min + vnum1 + c + spin_skip]
                    init_energy = energy_k[v_min + v + spin_skip]
                    energy_diff = final_energy - init_energy
                    bse_mat[s0][mat_idx, mat_idx] = energy_diff

            # No longer needed
            del energy_k, energy_kq

            # Off-diagonal elements:
            for l1, l2 in product(range(nk), range(nk)):
                kp_i = nk*l1 + l2
                kkp_1bz = [m1, m2, l1, l2]
                for s0 in range(self.n_spins):
                    # Read out precalculated scattering matrix
                    k_str = self.four_point_str % tuple(kkp_1bz)
                    s_str = self.spin_str % s0
                    scatter_int = np.array(g[s_str][k_str]['mat_elems'])

                    k_skip = k_i*block_skip
                    kp_skip = kp_i*block_skip

                    if selective or energy_cutoff_bool:
                        k_skip = blocks[s0][k_i]
                        kp_skip = blocks[s0][kp_i]

                    vnum1, cnum1 = self.get_number_conduction_valence_bands(
                        k_i, s0
                    )
                    vnum2, cnum2 = self.get_number_conduction_valence_bands(
                        kp_i, s0
                    )
                    if energy_cutoff_bool:
                        k_str = '(%d,%d)' % (m1, m2)
                        kp_str = '(%d,%d)' % (l1, l2)

                        v_min = np.array(
                            g['cutoff_bands'][s_str][k_str]['v_min']
                        )
                        v_min_p = np.array(
                            g['cutoff_bands'][s_str][kp_str]['v_min']
                        )
                        vnum1 = vnum1 - int(v_min)
                        vnum2 = vnum2 - int(v_min_p)

                        cnum1 = int(np.array(
                            g['cutoff_bands'][s_str][k_str]['c_max']
                        )) + 1
                        cnum2 = int(np.array(
                            g['cutoff_bands'][s_str][kp_str]['c_max']
                        )) + 1

                    for c1, c2 in product(range(cnum1), range(cnum2)):
                        for v1, v2 in product(range(vnum1), range(vnum2)):
                            fin_idx = k_skip + c1*vnum1 + v1
                            init_idx = kp_skip + c2*vnum2 + v2

                            sc_idx = c1*cnum2 + c2, v2*vnum1 + v1
                            int_term = scatter_int[sc_idx[0], sc_idx[1]]

                            # Note the sign of the direct integral
                            bse_mat[s0][fin_idx, init_idx] -= int_term

        g.close()
        # Decide whether to solve or output the hamiltonian to be
        # diagonalised elsewhere
        if solve:
            bse_eigsys = [napla.eigh(bse_mat[i]) for i in range(self.n_spins)]
        else:
            bse_eigsys = bse_mat

        return bse_eigsys

    def terminate_storage_usage(self):
        """Terminate the current usage session of the HDF5 file."""
        self.file_storage.close()

    def get_vector_modulo_cell(self, vector, macro_cell=False):
        """
        :return:
        """
        cell_factor = self.n_k if macro_cell else 1
        dot_divider = 2*np.pi*cell_factor
        n1 = (np.dot(vector, self.b1)/dot_divider)//1
        n2 = (np.dot(vector, self.b2)/dot_divider)//1
        vector_modulo_cell = vector - cell_factor*(n1*self.a1 + n2*self.a2)
        return vector_modulo_cell

    def create_fourier_matrix(self, pos, pos_idx, kdiff):
        fmatrix = np.array([
            [self.fourier_int((i, j), pos, pos_idx, kdiff)
             for j in range(self.n_atoms)] for i in range(self.n_atoms)
        ], dtype=complex)/self.n_k**2
        return fmatrix

    def get_eh_int(self, nat, nk, pos_list):
        """
        Get electron hole interaction matrix for a matrix of points
        :param nat: Number of atoms in the cell
        :param nk: nk X nk k grid.
        :param pos_list: list of positions of the periodic supercell points.
        :return:
        """
        b1, b2 = self.b1/nk, self.b2/nk
        nk2 = nk**2

        if nk == 1:
            # Only need a single point to calculate the fourier transform
            eh_int = list(np.zeros((1, nat, nat), dtype=complex))
            pos_idx = 0
            pos = pos_list[pos_idx]
            mat_term = self.create_fourier_matrix(pos=pos,
                                                  pos_idx=pos_idx,
                                                  kdiff=None)

            eh_int[0] += mat_term
        else:
            # Considering all differences between possible k-points
            eh_int = list(np.zeros((4*nk2, nat, nat), dtype=complex))
            for ml1, ml2 in product(range(2*nk), range(2*nk)):
                kdiff = (ml1 - nk)*b1 + (ml2 - nk)*b2
                for r1, r2 in product(range(nk), range(nk)):
                    pos_idx = r1*nk + r2
                    pos = pos_list[r1*nk + r2]
                    mat_term = self.create_fourier_matrix(pos=pos,
                                                          pos_idx=pos_idx,
                                                          kdiff=kdiff)
                    eh_int[ml1*2*nk + ml2] += mat_term
            # eh_int = list(np.zeros((nk2, nat, nat), dtype=complex))
            # for ml1, ml2 in product(range(nk), range(nk)):
            #     kdiff = ml1*b1 + ml2*b2
            #     for r1, r2 in product(range(nk), range(nk)):
            #         pos_idx = r1*nk + r2
            #         pos = pos_list[r1*nk + r2]
            #         mat_term = self.create_fourier_matrix(pos=pos,
            #                                               pos_idx=pos_idx,
            #                                               kdiff=kdiff)
            #         eh_int[ml1*nk + ml2] += mat_term
        return eh_int

    def get_vector_diff_modulo_cell(self, i, j):
        """
        Get separation between two points in the unit cell, modulo'd by
        lattice vectors so that the separation vector is within the unit cell.
        :param i: First index of atomic position in self.motif
        :param j: Second index of atomic position in self.motif
        :return:
        """
        tdiff = self.motif_vectors[i] - self.motif_vectors[j]
        n1 = (np.dot(tdiff, self.b1)/2/np.pi)//1
        n2 = (np.dot(tdiff, self.b2)/2/np.pi)//1
        tij = tdiff - n1*self.a1 - n2*self.a2
        return tij

    def fourier_int(self, idx_pair, pos, pos_idx, kpt):
        """
        Get the Lattice fourier transform of the interaction potential.
        :param pos: position of the cell in the supercell grid
        :param i: ith index of atom in cell
        :param j: jth index of atom in cell
        :param kpt: k point for the exponential
        :return:
        """
        i, j = idx_pair
        if self.cell_wise_centering:
            # Current method (change bool to True to keep this way)
            tij = self.get_vector_diff_modulo_cell(i, j)
            xij_1, xij_2 = recentre_continuous(tij, b1=self.b1, b2=self.b2)
            tij_cent = xij_1*self.a1 + xij_2*self.a2
            full_tij = pos + tij_cent
            radius = napla.norm(full_tij)
        else:
            # New method (change bool to False to try this way)
            tij = self.motif_vectors[i] - self.motif_vectors[j]
            full_pos = pos + tij
            full_pos_recentered = self.get_vector_modulo_cell(vector=full_pos,
                                                              macro_cell=True)
            xij_1, xij_2 = recentre_continuous(full_pos_recentered,
                                               b1=self.b1*self.n_k,
                                               b2=self.b2*self.n_k)
            full_tij = xij_1*self.a1*self.n_k + xij_2*self.a2*self.n_k
            radius = napla.norm(full_tij)
        fourier_term = interaction_potential(radius,
                                             self.potential_name,
                                             self.interaction_args,
                                             self.trunc_alat)
        if kpt is not None:
            fourier_term *= cplx_exp_dot(kpt, full_tij)

        return fourier_term

    def get_matrix_dim_and_block_starts(self, element_storage):
        """
        Get matrix dimensions.
        :param split:
        :return: matrix_dimension, block_skips
        """
        n_val, n_con, nk2 = self.n_val, self.n_con, self.n_k**2
        f = self.file_storage
        valcon = n_val*n_con
        mat_dim = [valcon*nk2]*self.n_spins
        block_starts, cumul_position, cumul_position_split = None, 0, [0, 0]
        # Make cumulative positions of elements in matrix to navigate through
        # bse matrix with kpoint blocks that have different number of bands

        one_point_str = self.one_point_str
        selective_bool = (self.selective_mode or
                          bool(element_storage['use_energy_cutoff']))
        if selective_bool:
            if self.n_spins == 1:
                blocks = []
                for idx in range(len(self.k_grid)):
                    v_num, c_num = n_val, n_con
                    if self.selective_mode:
                        k_str = one_point_str % idx
                        v_num_h5 = f['band_edges'][k_str]['vb_num']
                        c_num_h5 = f['band_edges'][k_str]['cb_num']
                        v_num = int(np.array(v_num_h5))
                        c_num = int(np.array(c_num_h5))
                    if bool(element_storage['use_energy_cutoff']):
                        xs = '(%d,%d)' % (idx % self.n_k, idx // self.n_k)
                        v_min = int(np.array(
                            element_storage['cutoff_bands']['s0'][xs]['v_min']
                        ))
                        v_num = v_num - v_min
                        c_max = int(np.array(
                            element_storage['cutoff_bands']['s0'][xs]['c_max']
                        ))
                        c_num = c_max + 1
                    blocks.append(cumul_position)
                    cumul_position += v_num*c_num
                # No spin-split system
                mat_dim = [cumul_position]
                block_starts = [blocks]
            else:
                blocks = [[], []]
                for idx in range(len(self.k_grid)):
                    v_num, c_num = [n_val, n_val], [n_con, n_con]
                    if self.selective_mode:
                        k_str = one_point_str % idx
                        v_num_h5 = f['band_edges'][k_str]['vb_num']
                        c_num_h5 = f['band_edges'][k_str]['cb_num']
                        v_num = list(np.array(v_num_h5))
                        c_num = list(np.array(c_num_h5))
                    if bool(element_storage['use_energy_cutoff']):
                        xs = '(%d,%d)' % (idx % self.n_k, idx // self.n_k)
                        v_min_0 = np.array(
                            element_storage['cutoff_bands']['s0'][xs]['v_min']
                        )
                        v_min_1 = np.array(
                            element_storage['cutoff_bands']['s1'][xs]['v_min']
                        )
                        c_max_0 = np.array(
                            element_storage['cutoff_bands']['s0'][xs]['c_max']
                        )
                        c_max_1 = np.array(
                            element_storage['cutoff_bands']['s1'][xs]['c_max']
                        )
                        v_num = [v_num[0] - v_min_0, v_num[1] - v_min_1]
                        c_num = [c_max_0 + 1, c_max_1 + 1]
                    for s0 in range(2):
                        blocks[s0].append(cumul_position_split[s0])
                        cumul_position_split[s0] += v_num[s0]*c_num[s0]

                mat_dim = cumul_position_split
                block_starts = blocks

        return mat_dim, block_starts

    def get_number_conduction_valence_bands(self, k_idx, s0):
        """
        Extract number of conduction and valence bands for a particular k point
        index.
        :param k_idx: index of the k point in self.k_grid
        :param s0: spin index (if a spinful model)
        :return: vb_num, cb_num
        """
        one_point_str = self.one_point_str
        f = self.file_storage
        if self.selective_mode:
            cb_num = f['band_edges'][one_point_str % k_idx]['cb_num']
            vb_num = f['band_edges'][one_point_str % k_idx]['vb_num']
            if self.n_spins == 2:
                cb_num = list(cb_num)[s0]
                vb_num = list(vb_num)[s0]
            else:
                cb_num = int(np.array(cb_num))
                vb_num = int(np.array(vb_num))
        else:
            vb_num, cb_num = self.n_val, self.n_con

        return vb_num, cb_num

    def get_diag_eigvals(self, k_idx, kq_idx, q_crys, direct):
        """
        Get the diagonal elements of the bse matrix,
        :param k_idx: First k point index with respect to self.k_grid
        :param kq_idx: Second k_point index with respect to self.k_grid
        :param q_crys: Crystal momentum of photon
        :param direct: Status of a direct excitation calculation
        :return:
        """
        h5_file = self.file_storage
        energy_k = np.array(h5_file['eigensystem']['eigenvalues'][k_idx])
        if direct or (q_crys[0] == 0 and q_crys[1] == 0):
            # This doesn't create another instance
            energy_kq = energy_k
        else:
            energy_kq = np.array(h5_file['eigensystem']['eigenvalues'][kq_idx])

        return energy_k, energy_kq

    def read_eigenvectors(self, idx_pair):
        """
        Extract eigenvectors from a certain set of rows.
        :param idx_pair: Indices to book-end the range of eigenvectors to
                         extract from
        :return:
        """
        i1, i2 = idx_pair[0], idx_pair[1]
        eigvecs = np.array(
            self.file_storage['eigensystem']['eigenvalues'][i1:i2, :]
        )
        return eigvecs

    def treat_phase_of_eigenvectors(self, eigvecs):
        """
        STILL NEEDS TESTING: Fix global phase of all eigenvectors in dataset
        :param eigvecs: set of Eigenvectors
        :return:
        """
        eigcopy = np.array(eigvecs)
        valcon = self.n_val + self.n_con
        for idx in range(len(self.k_grid)):
            k_shift = idx*self.n_orbs
            if self.n_spins == 1:
                bandnum = self.get_number_conduction_valence_bands(idx, 0)
                for jdx in range(bandnum[0] + bandnum[1]):
                    p, q = k_shift, k_shift + valcon
                    eigcopy[p:q, jdx] = fix_consistent_gauge(eigvecs[p:q, jdx])

        return eigcopy

    def get_preprocessed_eigenvectors(self):
        """
        Combine real and imaginary parts of eigenvectors into an array,
        if not complex.
        :param eigenvectors:
        :return:
        """
        f = self.file_storage
        if self.is_complex:
            eigenvectors = np.array(f['eigensystem']['eigenvectors'])
        else:
            eigvecs_real = np.array(f['eigensystem']['eigenvectors_real'])
            eigvecs_imag = np.array(f['eigensystem']['eigenvectors_imag'])
            eigenvectors = eigvecs_real + 1j*eigvecs_imag

        if self.convention == 2:
            eigenvectors = convert_all_eigenvectors(eigenvectors,
                                                    self.k_grid,
                                                    self.motif_vectors,
                                                    self.orb_pattern,
                                                    self.n_orbs,
                                                    bool(self.n_spins == 2))
        # elif self.convention == 1:
        elif not self.convention == 1:
            # eigenvectors = unconvert_all_eigenvectors(eigenvectors,
            #                                           self.k_grid,
            #                                           self.motif_vectors,
            #                                           self.orb_pattern,
            #                                           self.n_orbs,
            #                                           bool(self.n_spins == 2))
            raise ValueError("Convention should be 1 or 2.")

        eigenvalues = np.array(f['eigensystem']['eigenvalues'])
        eigenvectors = orthogonalize_eigenvecs(eigenvalues, eigenvectors)
        return eigenvectors

    def get_cutoff_band_min_maxes(self, energy_cutoff):
        """
        Calculate an array of all minimum valence band indices and maximum
        conduction band indices for each k point index and spin index
        (if applicable).
        @param energy_cutoff: Value with which to deteremine required bands.
        @return:
        """
        eigvals = np.array(self.file_storage['eigensystem']['eigenvalues'])
        cutoff_band_min_maxes = np.zeros((self.n_spins, self.n_k**2, 2))

        for s0 in range(self.n_spins):
            for m1, m2 in product(range(self.n_k), range(self.n_k)):
                k_idx = m1*self.n_k + m2
                v_num, c_num = self.get_number_conduction_valence_bands(k_idx,
                                                                        s0)
                if energy_cutoff is not None:
                    eigvals_k = eigvals[k_idx]
                    idx_1 = s0*(len(eigvals_k) - v_num - c_num)
                    idx_2 = len(eigvals_k) if bool(s0) else v_num + c_num
                    eigvals_k_s0 = eigvals_k[idx_1:idx_2]
                    c_max, v_min = self.get_band_extrema(eigvals_k_s0,
                                                         energy_cutoff,
                                                         v_num)
                    cutoff_band_min_maxes[s0][k_idx][0] = v_min
                    cutoff_band_min_maxes[s0][k_idx][1] = c_max - v_num
                else:
                    cutoff_band_min_maxes[s0][k_idx][0] = 0
                    cutoff_band_min_maxes[s0][k_idx][1] = c_num - 1

        return cutoff_band_min_maxes

    @staticmethod
    def get_band_extrema(eigenvalues, energy_cutoff, edge_index):
        """
        Given an energy cutoff, finds the band indices that given the minimum
        valence band within the cutoff below the conduction band, and the
        maximum conduction band within a cutoff above the valence band.
        @param eigenvalues: list of eigenvalues
        @param energy_cutoff: energy to decide which bands we require
        @param edge_index: index of the conduction band
        @return:
        """

        max_energy = eigenvalues[edge_index - 1] + energy_cutoff
        min_energy = eigenvalues[edge_index] - energy_cutoff

        try:
            cb_max = list(eigenvalues <= max_energy).index(False) - 1
        except ValueError:
            cb_max = len(eigenvalues) - 1

        try:
            vb_min = list(eigenvalues >= min_energy).index(True)
        except ValueError:
            vb_min = 0

        return cb_max, vb_min

    def get_cutoff_bands_info(self, energy_cutoff):
        """
        Obtain the information about the bands used in the cutoff
        (or not cutoff) calculation
        @param energy_cutoff:
        @return: cutoff_bands_info
        """
        cutoff_bands_info = [[] for i in range(self.n_spins)]
        cutoff_band_min_maxes = self.get_cutoff_band_min_maxes(energy_cutoff)
        for s0 in range(self.n_spins):
            for idx, kpt in enumerate(self.k_grid):
                nval, ncon = self.get_number_conduction_valence_bands(s0, idx)
                vmin_cmax = cutoff_band_min_maxes[0][idx]
                avail_vb = range(int(vmin_cmax[0]), nval)
                avail_cb = range(int(vmin_cmax[1]) + 1)
                # Should ensure that this ordering matches that of the
                # Hamiltonian construction
                kpt_dict = {}
                combos = [(c, v) for c, v in product(avail_cb, avail_vb)]
                kpt_dict['transitions'] = combos
                kpt_dict['k_point'] = kpt
                cutoff_bands_info[s0].append([idx, kpt_dict])
        return cutoff_bands_info
