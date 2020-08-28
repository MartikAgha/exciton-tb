import os
import unittest

import numpy as np

from exciton_tb.exciton_tb import ExcitonTB


class TestExcitonTB(unittest.TestCase):

    test_input_folder = 'tests/testing_data'
    test_input_template = 'test_sample_%s.hdf5'
    dp = 6

    def setUp(self):
        self.init_interactions = ['keldysn', 'yukawa', 'coulomb']
        self.hdf5_inputs = ['test_sample_1.hdf5',
                            'test_sample_2.hdf5',
                            'test_sample_3.hdf5']
        self.init_properties = {
            '1': {
                'crystal': {
                    'alat': 3.19, 'avecs_a1': np.array([3.19, 0.0]),
                    'avecs_a2': np.array([1.595, -2.76262104]),
                    'motif': np.array([0, 0]), 'n_atoms': 1, 'n_orbs': 6,
                    'orb_pattern': np.array([6])
                },
                'eigensystem': {
                    'convention': 1, 'is_complex': True,
                    'k_grid': np.array([[0., 0.],
                                        [0., -0.75811886],
                                        [0., -1.51623771],
                                        [0.65655019,  0.37905943],
                                        [0.65655019, -0.37905943],
                                        [0.65655019, -1.13717828],
                                        [1.31310038,  0.75811886],
                                        [1.31310038,  0.],
                                        [1.31310038, -0.75811886]]),
                    'n_con': 4, 'n_val': 2, 'n_k': 3, 'n_spins': 1
                }
            },
            '2': {
                'crystal': {
                    'alat': 9.57, 'avecs_a1': np.array([9.57, 0.0]),
                    'avecs_a2': np.array([4.785, -8.28786311]),
                    'motif': np.array([[0., -0.],
                                       [3.19, -0.],
                                       [6.38, -0.],
                                       [1.595, -2.76262104],
                                       [4.785, -2.76262104],
                                       [7.975, -2.76262104],
                                       [3.19, -5.52524208],
                                       [6.38, -5.52524208],
                                       [9.57, -5.52524208]]),
                    'n_atoms': 9, 'n_orbs': 54,
                    'orb_pattern': np.array([6])
                },
                'eigensystem': {
                    'convention': 1, 'is_complex': True,
                    'k_grid': np.array([[0., 0.]]),
                    'n_con': 36, 'n_val': 18, 'n_k': 1, 'n_spins': 1
                }
            },
            '3': {
                'crystal': {
                    'alat': 9.57, 'avecs_a1': np.array([9.57, 0.0]),
                    'avecs_a2': np.array([4.785, -8.28786311]),
                    'motif': np.array([[0., -0.],
                                       [3.19, -0.],
                                       [6.38, -0.],
                                       [1.595, -2.76262104],
                                       [4.785, -2.76262104],
                                       [7.975, -2.76262104],
                                       [3.19, -5.52524208],
                                       [6.38, -5.52524208],
                                       [9.57, -5.52524208]]),
                    'n_atoms': 9, 'n_orbs': 54,
                    'orb_pattern': np.array([6])
                },
                'eigensystem': {
                    'convention': 1, 'is_complex': True,
                    'k_grid': np.array([[0., 0.],
                                        [0., -0.25270629],
                                        [0., -0.50541257],
                                        [0.21885006,  0.12635314],
                                        [0.21885006, -0.12635314],
                                        [0.21885006, -0.37905943],
                                        [0.43770013,  0.25270629],
                                        [0.43770013,  0.],
                                        [0.43770013, -0.25270629]]),
                    'n_con': 36, 'n_val': 18, 'n_k': 3, 'n_spins': 1
                }
            }
        }

    def assertListAlmostEqual(self, list1, list2, places):
        self.assertEqual(len(list1), len(list2))
        for elem_a, elem_b in zip(list1, list2):
            self.assertAlmostEqual(elem_a, elem_b, places=places)

    def test_exciton_tb_init(self):

        for test_tag, test_dict in self.init_properties.items():
            test_file = self.test_input_template % test_tag
            test_path = os.path.join(self.test_input_folder, test_file)

            extb = ExcitonTB(test_path)

            # Integer/bool equality assertions
            self.assertEqual(test_dict['crystal']['n_atoms'], extb.n_atoms)
            self.assertEqual(test_dict['crystal']['n_orbs'], extb.n_orbs)
            self.assertEqual(test_dict['eigensystem']['n_k'], extb.n_k)
            self.assertEqual(test_dict['eigensystem']['n_con'], extb.n_con)
            self.assertEqual(test_dict['eigensystem']['n_val'], extb.n_val)
            self.assertEqual(test_dict['eigensystem']['n_spins'],
                             extb.n_spins)
            self.assertEqual(test_dict['eigensystem']['convention'],
                             extb.convention)
            self.assertEqual(test_dict['eigensystem']['is_complex'],
                             extb.is_complex)

            # Float equality assertions
            self.assertAlmostEqual(test_dict['crystal']['alat'],
                                   extb.alat,
                                   places=self.dp)
            # List properties to check equality with.
            self.assertListAlmostEqual(test_dict['crystal']['motif'].ravel(),
                                       extb.motif_vectors.ravel(),
                                       places=self.dp)
            self.assertListAlmostEqual(
                test_dict['eigensystem']['k_grid'].ravel(),
                extb.k_grid.ravel(),
                places=self.dp
            )
            self.assertListAlmostEqual(test_dict['crystal']['avecs_a1'],
                                       extb.a1,
                                       places=self.dp)
            self.assertListAlmostEqual(test_dict['crystal']['avecs_a2'],
                                       extb.a2,
                                       places=self.dp)

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