import unittest

from h5py import File

from exciton_tb.exciton_tb import ExcitonTB


class TestExcitonTB(unittest.TestCase):

    def setUp(self):
        init_interactions = ['keldysn', 'yukawa', 'coulomb']


    def test_exciton_tb_init(self):
        raise NotImplementedError()

    def test_hdf5_storage_creation(self):
        raise NotImplementedError()

    def test_vector_modulo_cell(self):
        raise NotImplementedError()

    def test_fourier_matrix_creation(self):
        raise NotImplementedError()

    def test_electron_hole_interaction(self):
        raise NotImplementedError()


if __name__ == '__main__':
    unittest.main()