#!/usr/bin/python3
from itertools import product

import scipy as sp
import numpy as np
import h5py as hp
import numpy.linalg as napla

from .exciton_tools import get_complex_zeros, tr_keld, cplx_exp_dot, \
                           get_cumulative_positions, reduced_tb_vec, \
                           recentre_continuous, get_supercell_positions

class ExcitonTB:

    one_point_str = 'k(%d)'
    two_point_str = 'k(%d,%d)'
    four_point_str = 'k(%d,%d,%d,%d)'

    hermitian_msg = "{} != {} , Not hermitian"
    hermitian_rounding_dp = 8
    round_dp = 7

    default_args = {'keldysh_r0': 33.875,
                    'substrate_dielectric': 3.8}

    def __init__(self, hdf5_input, potential='keldysh', args=None):
        """
        Exciton calculator for tight-binding models.
        :param hdf5_input: name of hdf5 input that contains crystal information
                           and eigensystems
        :param potential: type of potential used (only keldysh for now)
        :param args: potential parameters
        """
        self.potential = potential
        self.interaction_args = self.default_args if args is None else args

        self.file_storage = hp.File(hdf5_input, 'r')
        # Split is always True for this calculation
        self.alat = float(np.array(self.file_storage['crystal']['alat']))
        self.a1 = np.array(self.file_storage['crystal']['avecs']['a1'])
        self.a2 = np.array(self.file_storage['crystal']['avecs']['a2'])
        self.b1 = np.array(self.file_storage['crystal']['gvecs']['b1'])
        self.b2 = np.array(self.file_storage['crystal']['gvecs']['b2'])

        # FOR NOW: Assume one orbital per atom
        self.n_atoms = int(np.array(self.file_storage['crystal']['n_atoms']))
        self.n_orbs = int(np.array(self.file_storage['crystal']['n_orbs']))
        self.n_k = int(np.array(self.file_storage['eigensystem']['n_k']))
        self.n_val = int(np.array(self.file_storage['eigensystem']['n_val']))
        self.n_con = int(np.array(self.file_storage['eigensystem']['n_con']))
        self.k_grid = np.array(self.file_storage['eigensystem']['k_grid'])
        self.r_grid = get_supercell_positions(self.a1, self.a2, self.n_k)

        self.trunc_alat = float(np.array(self.file_storage['crystal']['trunc_alat']))
        self.motif_vectors = np.array(self.file_storage['crystal']['motif'])
        self.centre_point = np.array(self.file_storage['crystal']['centre'])
        orb_pattern = list(self.file_storage['crystal']['orb_pattern'])
        self.cumulative_positions = get_cumulative_positions(orb_pattern,
                                                             self.n_orbs)

        self.selective_mode = bool(np.array(self.file_storage['selective']))
        if self.selective_mode:
            self.num_states = int(np.array(
                self.file_storage['band_edges']['num_states']
            ))

    def terminate_storage_usage(self):
        """Terminate the current usage session of the HDF5 file."""
        self.file_storage.close()

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
            eh_int = list(sp.zeros((1, nat, nat), dtype=complex))
            pos = pos_list[0]
            mat_term = np.array([
                [self.fourier_keld(pos, i, j, None) for j in range(nat)]
                for i in range(nat)
            ], dtype=complex)

            eh_int[0] += mat_term
        else:
            # Considering all differences between possible k-points
            eh_int = list(sp.zeros((4*nk2, nat, nat), dtype=complex))
            for ml1, ml2 in product(range(2*nk), range(2*nk)):
                kdiff = (ml1 - nk)*b1 + (ml2 - nk)*b2
                for r1, r2 in product(range(nk), range(nk)):
                    pos = pos_list[r1*nk + r2]
                    mat_term = np.array([
                        [self.fourier_keld(pos, i, j, kdiff)/nk2
                         for j in range(nat)]
                        for i in range(nat)
                    ], dtype=complex)
                    eh_int[ml1*2*nk + ml2] += mat_term
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
        n1 = np.round((np.dot(tdiff, self.b1)/2/np.pi), self.round_dp)//1
        n2 = np.round((np.dot(tdiff, self.b2)/2/np.pi), self.round_dp)//1
        tij = tdiff - n1*self.a1 - n2*self.a2
        return tij

    def fourier_keld(self, pos, i, j, kpt):
        """
        Keldysh lattice fourier transform
        :param pos: position of the cell in the supercell grid
        :param i: ith index of atom in cell
        :param j: jth index of atom in cell
        :param kpt: k point for the exponential
        :return:
        """
        radius_0 = self.interaction_args['keldysh_r0']
        alat = self.trunc_alat
        ep12 = 0.5*(1 + self.interaction_args['substrate_dielectric'])

        tij = self.get_vector_diff_modulo_cell(i, j)
        xij_1, xij_2 = recentre_continuous(tij, b1=self.b1, b2=self.b2)
        tij_cent = xij_1*self.a1 + xij_2*self.a2
        r_vec = pos + tij_cent

        fourier_term = tr_keld(napla.norm(r_vec), radius_0, ep12, alat)
        if kpt is not None:
            fourier_term *= cplx_exp_dot(kpt, pos)

        return fourier_term

    def get_matrix_dim_and_block_starts(self, split=True):
        """
        Get matrix dimensions.
        :param split:
        :return: matrix_dimension, block_skips
        """
        n_val, n_con, nk2 = self.n_val, self.n_con, self.n_k**2
        f = self.file_storage
        valcon = n_val*n_con
        mat_dim = [valcon*nk2, valcon*nk2] if split else 2*valcon*nk2
        block_starts, cumul_position, cumul_position_split = None, 0, [0, 0]
        # Make cumulative positions of elements in matrix to navigate

        one_point_str = self.one_point_str
        if self.selective_mode:
            blocks = []
            blocks_split = [[], []]
            for idx in range(len(self.k_grid)):
                v_num = list(f['band_edges'][one_point_str % idx]['vb_num'])
                c_num = list(f['band_edges'][one_point_str % idx]['cb_num'])

                for s0 in range(2):
                    if split:
                        blocks_split[s0].append(cumul_position_split[s0])
                        cumul_position_split[s0] += v_num[s0]*c_num[s0]
                    else:
                        blocks.append(cumul_position)
                        cumul_position += v_num[s0]*c_num[s0]

            if split:
                mat_dim = cumul_position_split
                block_starts = blocks_split
            else:
                mat_dim = cumul_position
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
            cb_num = list(f['band_edges'][one_point_str % k_idx]['cb_num'])[s0]
            vb_num = list(f['band_edges'][one_point_str % k_idx]['vb_num'])[s0]
        else:
            vb_num, cb_num = self.n_val, self.n_con

        return vb_num, cb_num

    def create_matrix_element_hdf5(self, element_storage):
        """
        Create matrix elements for the direct coulomb interaction
        :param element_storage: name for the hdf5 storage for matrix elements.
        :return:
        """
        self.element_storage_name = element_storage
        """Create store of eigenvalues, matrix elements and attributes."""
        # Acquire atomic structure parameters.
        nat = self.n_atoms
        norb = self.n_orbs
        nk, nk2 = self.n_k, self.n_k**2

        # Containers for outputted data
        pos_list = self.r_grid

        with hp.File(element_storage, 'w') as f:
            eigvecs = np.array(self.file_storage['eigensystem']['eigenvectors'])
            eh_int = self.get_eh_int(nat, nk, pos_list)
            f['VkM'] = eh_int
            nk_shift = nk if nk == 1 else 2*nk

            num_states = self.num_states if self.selective_mode else norb

            for s0 in range(2):
                # Two separate sets for up and down spin for non-exchange
                # Assuming a Gamma-point calculation for now in supercell
                spins = f.create_group('s{}'.format(s0))

                for m1, m2 in product(range(nk), range(nk)):
                    for l1, l2 in product(range(nk), range(nk)):
                        # Indices to navigate through k,k' data
                        if nk > 1:
                            ml1, ml2 = m1 - l1 + nk, m2 - l2 + nk

                        else:
                            ml1, ml2 = (m1 - l1) % nk, (m2 - l2) % nk

                        ki_m, ki_l = m1*nk + m2, l1*nk + l2

                        bandnum_m = self.get_number_conduction_valence_bands(
                            ki_m, s0
                        )
                        v_num_m, c_num_m = bandnum_m

                        bandnum_l = self.get_number_conduction_valence_bands(
                            ki_l, s0
                        )
                        v_num_l, c_num_l = bandnum_l

                        valence = np.zeros((nat, v_num_m*v_num_l),
                                           dtype='complex')
                        conduction = np.zeros((nat, c_num_m*c_num_l),
                                              dtype='complex')

                        m_shf, l_shf = ki_m*2*num_states, ki_l*2*num_states
                        id_string = self.four_point_str % (m1, m2, l1, l2)
                        kpts = spins.create_group(id_string)
                        e_m, e_l = m_shf + s0*num_states, l_shf + s0*num_states

                        vb_iter = product(range(v_num_m), range(v_num_l))
                        for v1, v2 in vb_iter:
                            # for v1, k' and v2, k find matrix element
                            # This is reversed compared to conduction band
                            vm_i = v2*v_num_m + v1
                            valence[:, vm_i] = reduced_tb_vec(
                                eigvecs[e_l:e_l + num_states, v2],
                                eigvecs[e_m:e_m + num_states, v1],
                                nat,
                                self.cumulative_positions
                            )
                        cb_iter = product(range(c_num_m), range(c_num_l))
                        for c1, c2 in cb_iter:
                            # for c1, k and c2, k' find matrix element
                            cm_i = c1*c_num_l + c2
                            conduction[:, cm_i] = reduced_tb_vec(
                                eigvecs[e_m:e_m + norb, v_num_m + c1],
                                eigvecs[e_l:e_l + norb, v_num_l + c2],
                                nat,
                                self.cumulative_positions
                            )
                        elem_shape = (c_num_m*c_num_l, v_num_m*v_num_l)
                        mmm = kpts.create_dataset(name='mat_elems',
                                                  shape=elem_shape,
                                                  dtype='complex')

                        mmm[:] = np.dot(np.array(conduction).T,
                                        np.dot(eh_int[ml1*nk_shift + ml2],
                                               np.array(valence)))

                        del conduction, valence

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

    def read_scatter_matrix(self, element_storage, s_str, k_str):
        return np.array(element_storage[s_str][k_str]['mat_elems'])

    def get_bse_eigensystem_direct(self, matrix_element_storage, solve=True):
        """
        Construct the Hamiltonian (and solve it if necessary) for the
        Bethe-Salpeter Equation
        :param matrix_element_storage: HDF5 Storage for the matrix elements.
        :param solve: Set to True to find the eigenvalues, otherwise will
                      output the matrix.
        :return:
        """
        g = hp.File(matrix_element_storage, 'r')
        nk, nk2 = self.n_k, self.n_k**2
        n_val, n_con = self.n_val, self.n_con
        selective = self.selective_mode

        mat_dim, blocks = self.get_matrix_dim_and_block_starts(split=True)
        block_skip = n_con*n_val
        spin_shift = n_con + n_val
        bse_mat = [get_complex_zeros(mat_dim[i]) for i in range(2)]
        q_zero = np.array([0, 0])

        for m1, m2 in product(range(nk), range(nk)):
            k_i = nk*m1 + m2


            # Diagonal elements: Use cb/vb energy differences
            energy_k, energy_kq = self.get_diag_eigvals(k_idx=k_i,
                                                        kq_idx=k_i,
                                                        q_crys=q_zero,
                                                        direct=True)

            for s0 in range(2):
                k_skip = blocks[s0][k_i] if selective else k_i*block_skip
                s_skip = self.num_states if selective else spin_shift
                vnum1, cnum1 = self.get_number_conduction_valence_bands(
                    k_i, s0
                )

                for c, v in product(range(cnum1), range(vnum1)):
                    mat_idx = k_skip + c*vnum1 + v
                    final_energy = energy_kq[vnum1 + c + s0*s_skip]
                    init_energy = energy_k[v + s0*s_skip]
                    energy_diff = final_energy - init_energy
                    bse_mat[s0][mat_idx, mat_idx] = energy_diff

            # No longer needed
            del energy_k, energy_kq

            # Off-diagonal elements:
            for l1, l2 in product(range(nk), range(nk)):
                kp_i = nk*l1 + l2
                kkp_1bz = [m1, m2, l1, l2]
                for s0 in range(2):
                    # Read out precalculated scattering matrix
                    scatter_int = self.read_scatter_matrix(
                        element_storage=g,
                        s_str='s%d' % s0,
                        k_str=self.four_point_str % tuple(kkp_1bz)
                    )

                    k_skip = blocks[s0][k_i] if selective else k_i*block_skip
                    kp_skip = blocks[s0][kp_i] if selective else kp_i*block_skip

                    vnum1, cnum1 = self.get_number_conduction_valence_bands(
                        k_i, s0
                    )
                    vnum2, cnum2 = self.get_number_conduction_valence_bands(
                        kp_i, s0
                    )

                    for c1, c2 in product(range(cnum1), range(cnum2)):
                        for v1, v2 in product(range(vnum1), range(vnum2)):
                            fin_idx = k_skip + c1*vnum1 + v1
                            init_idx = kp_skip + c2*vnum2 + v2

                            sc_idx = c1*cnum2 + c2, v2*vnum1 + v1
                            int_term = scatter_int[sc_idx[0], sc_idx[1]]

                            # Note the sign of the direct integral
                            bse_mat[s0][fin_idx, init_idx] -= int_term

        # Decide whether to solve or output the hamiltonian to be
        # diagonalised elsewhere
        if solve:
            exciton_eigensystems = [napla.eigh(bse_mat[i]) for i in range(2)]
        else:
            exciton_eigensystems = bse_mat

        return exciton_eigensystems
