
import numpy as np
import h5py as hp


class ConductivityTB:

    polarisation_vectors = {'x': np.array([1, 0]),
                            'y': np.array([0, 1]),
                            'lh': np.array([1j, 1]),
                            'rh': np.array([-1j, 1])}

    def __init__(self, hdf5_input):
        """

        :param hdf5_input:
        """
        self.data_storage = hp.File(hdf5_input, 'r')

        # Remains open for the calculation, use self.terminate_storage_usage
        # when done using this.
        # Extract geometric information
        self.alat = float(np.array(self.data_storage['crystal']['alat']))
        self.a1 = np.array(self.data_storage['crystal']['avecs']['a1'])
        self.a2 = np.array(self.data_storage['crystal']['avecs']['a2'])
        self.b1 = np.array(self.data_storage['crystal']['gvecs']['b1'])
        self.b2 = np.array(self.data_storage['crystal']['gvecs']['b2'])
        self.motif_vectors = np.array(self.data_storage['crystal']['motif'])

        self.n_atoms = int(np.array(self.data_storage['crystal']['n_atoms']))
        self.n_orbs = int(np.array(self.data_storage['crystal']['n_orbs']))
        self.n_k = int(np.array(self.data_storage['eigensystem']['n_k']))
        self.n_val = int(np.array(self.data_storage['eigensystem']['n_val']))
        self.n_con = int(np.array(self.data_storage['eigensystem']['n_con']))
        self.n_spins = int(np.array(self.data_storage['eigensystem']['n_spins']))

        # Acrue real-space and reciprocal-space grid.
        self.k_grid = np.array(self.data_storage['eigensystem']['k_grid'])

    def terminate_hdf5_session(self):
        """Close HDF5 storage used for this class."""
        self.data_storage.close()

    def load_velocity_matrix(self, k_idx, s0):
        """
        Load velocity matrix for a specific k_points (and potentially spin).
        :param k_idx: idx of the k-point with respect to self.k_grid
        :param s0: spin index if spin-divided (i.e. n_spins == 2)
        :return: velocity_matrix
        """
        if self.n_spins == 1:
            vel_mat = np.array(self.data_storage['velocity_matrix'][k_idx])
        else:
            vel_mat = np.array(self.data_storage['velocity_matrix'][k_idx][s0])
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

    def serial_conductivity(self,
                            sigma=0.04,
                            direction='x',
                            freq_range=(0.0, 6.0, 1000),
                            interaction=True):
        if interaction:
            pass
        else:


