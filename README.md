Input file:

hdf5 format with subdivisions:

'crystal': details about the crystal including

        - 'alat': lattice parameter

        - 'avecs': list of lattice vectors

        - 'centre': some central vector of the cell

        - 'gvecs': list of reciprocal lattice vectors

        - 'motif': list of vectors for all positions in the cell

        - 'n_atoms': number of atoms in cell

        - 'n_orbs': total number of orbitals in the cell (not including spin)

        - 'orb_pattern': list of orbitals per atom repeat e.g. [3, 3, 5]

        - 'positions': Periodic supercell lattice vector list

'eigensystem': eigenvalue and eigenvector data

        - 'eigenvalues'

        - 'eigenvectors'

        - 'kgrid': list of reciprocal vectors that are used for the calculations

        - 'n_k': size of kgrid i.e. the grid will be nk times nk

        - 'n_con': number of conduction band states

        - 'n_val': number of valence band states

'selective': The number of bands to be used in the system are limited to avoid
using all transitions when calculating the excitons, and instead to use a
limited subset of bands. set to True or False.

'band_edges': If selective is True, the minvalence band and max conduction band

    - 'num_states': number of states included in each eigensystem

    - 'k(idx)' idx of the kpt in the k_grid:

            - 'vb_num': number of valence bands

            - 'cb_num': number of conduction bands