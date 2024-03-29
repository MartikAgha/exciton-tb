### Usage:

Given HDF5 input file 'h5_inp' (see below), use as follows
(while in 'src' directory):


    >>> from exciton_tb.exciton_tb import ExcitonTB

    >>> extb = ExcitonTB('/path/to/h5_inp')

    >>> extb.create_matrix_element_hdf5('name_of_matrix_element.hdf5')

    >>> bse_hamiltonian = extb.get_bse_eigensystem_direct(solve=False)

    >>> bse_eigensystem = extb.get_bse_eigensystem_direct(solve=True)



### Input file:

hdf5 format with subdivisions:

Resource for h5py usage: http://docs.h5py.org/en/stable/

- 'crystal': details about the crystal including

    - 'alat': lattice parameter

    - 'avecs': Real Space Lattice vectors
    
        - 'a1' : First lattice vector

        - 'a2' : Second lattice vector

    - 'gvecs': Reciprocal Space Lattice vectors
    
        - 'b1' : First reciprocal lattice vector

        - 'b2' : Second reciprocal lattice vector
        
    - 'motif': list of vectors for all positions in the cell

    - 'n_atoms': number of atoms in cell

    - 'n_spins': 1 to treat either spinless system or to not split the
                 system into two spins. 2 to construct bse_hamiltonians for
                 spin-up transitions and spin-down transitions.

    - 'n_orbs': total number of orbitals in the cell (not including spin)

    - 'orb_pattern': list of orbitals per atom repeat e.g. [3, 3, 5]

    - 'positions': Periodic supercell lattice vector list
    
    - 'trunc_alat': Distance to use in the potential near R=0

- 'eigensystem': eigenvalue and eigenvector data

    - 'convention': Either 1 or 2, ref. https://www.physics.rutgers.edu/pythtb/formalism.html

    - 'n_spins': Set to 1 if spin is either unpresent in the calculation,
                 or if the hamiltonian cannot be split into separable
                 blocks corresponding to different spins. Set to 2 if the
                 eigenvectors for different spins are calculated separately
                 and can be placed in eigenvectors as blocks half the size.

    - 'eigenvalues': Eigenvalues for each k point. Eigenvalues from the
                     same k point should exist in the same row in
                     numerical order, such that the shape of 'eigenvalues'
                     is (n_kpoints, n_states). If the eigensystem is
                     divided by spin (a spin-separable hamiltonian), then
                     the order of eigenvalues should have increasingly
                     ordered spin down eigenvalues first followed by
                     increasingly ordered spin up eigenvalues.

    - 'eigenvectors': Eigenvectors for each k point. Eigenvectors from
                      the same k-point should be in a block of size
                      (eigenvector_length, n_states), and these should be
                      stacked row-wise for different k points, such that
                      'eigenvectors' is of size
                      (n_kpoints*eigenvector_length, n_states). Note if
                      different k points have different numbers of states
                      associated with them, then the number of columns
                      should be the largest number of states at a single
                      k point, and for k points with less states, the
                      latter states should appear as zeros on the
                      right-most columns.

    - 'eigenvectors_real': Real part of the eigenvectors, see
                           'eigenvectors' for structural information

    - 'eigenvectors_imag': Imaginary part of the eigenvectors, see
                           'eigenvectors' for structural information

    - 'is_complex': True to treat the 'eigenvectors' as complex entities,
                    otherwise, set this to False and specify
                    'eigenvectors_real' and 'eigenvectors_imag'

    - 'k_grid': list of reciprocal vectors that are used for the calculations

    - 'n_k': size of kgrid i.e. the grid will be nk times nk

    - 'n_con': number of conduction band states.
               (Not needed if 'selective' is True)

    - 'n_val': number of valence band states
               (Not needed if 'selective' is True)

- 'selective': The number of bands to be used in the system are limited to avoid
using all transitions when calculating the excitons, and instead to use a
limited subset of bands. set to True or False.

- 'band_edges': If selective is True, the minvalence band and max conduction band

    - 'num_states': number of states included in each eigensystem

    - 'k(idx)' idx of the kpt in the k_grid. (Note if n_spins is 2, and we
                wish to treat the spins separately, the values of 'vb_num'
                and 'cb_num' will be a list of 2 values for each spin)

        - 'vb_num': number of valence bands

        - 'cb_num': number of conduction bands
  
- 'velocity_matrix': (Optional) If the conductivity of imaginary part of the 
                     dielectric function is required then the velocity matrices,
                     will be placed here. The velocity matrix must be calculated
                     at the same k points that are used in the calculation, and 
                     included in this file in the same order as the k points 
                     appear in 'eigensystem/k_grid'

    - 'k(idx)' Place velocity matrix here. idx of the kpt in the k_grid.  

