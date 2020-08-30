import os
import unittest

import numpy as np
from h5py import File

from exciton_tb.exciton_tb import ExcitonTB


class TestExcitonTB(unittest.TestCase):

    test_input_folder = 'tests/testing_data'
    test_input_template = 'test_sample_%s.hdf5'
    dp = 6
    element_dir = '../matrix_element_hdf5'
    element_name_template = 'matrix_element_{}_{}.hdf5'
    element_key = 'matrix_element_size'

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
        three_by_three_list = [
            'k(0,0,0,0)', 'k(0,0,0,1)', 'k(0,0,0,2)', 'k(0,0,1,0)',
            'k(0,0,1,1)', 'k(0,0,1,2)', 'k(0,0,2,0)', 'k(0,0,2,1)',
            'k(0,0,2,2)', 'k(0,1,0,0)', 'k(0,1,0,1)', 'k(0,1,0,2)',
            'k(0,1,1,0)', 'k(0,1,1,1)', 'k(0,1,1,2)', 'k(0,1,2,0)',
            'k(0,1,2,1)', 'k(0,1,2,2)', 'k(0,2,0,0)', 'k(0,2,0,1)',
            'k(0,2,0,2)', 'k(0,2,1,0)', 'k(0,2,1,1)', 'k(0,2,1,2)',
            'k(0,2,2,0)', 'k(0,2,2,1)', 'k(0,2,2,2)', 'k(1,0,0,0)',
            'k(1,0,0,1)', 'k(1,0,0,2)', 'k(1,0,1,0)', 'k(1,0,1,1)',
            'k(1,0,1,2)', 'k(1,0,2,0)', 'k(1,0,2,1)', 'k(1,0,2,2)',
            'k(1,1,0,0)', 'k(1,1,0,1)', 'k(1,1,0,2)', 'k(1,1,1,0)',
            'k(1,1,1,1)', 'k(1,1,1,2)', 'k(1,1,2,0)', 'k(1,1,2,1)',
            'k(1,1,2,2)', 'k(1,2,0,0)', 'k(1,2,0,1)', 'k(1,2,0,2)',
            'k(1,2,1,0)', 'k(1,2,1,1)', 'k(1,2,1,2)', 'k(1,2,2,0)',
            'k(1,2,2,1)', 'k(1,2,2,2)', 'k(2,0,0,0)', 'k(2,0,0,1)',
            'k(2,0,0,2)', 'k(2,0,1,0)', 'k(2,0,1,1)', 'k(2,0,1,2)',
            'k(2,0,2,0)', 'k(2,0,2,1)', 'k(2,0,2,2)', 'k(2,1,0,0)',
            'k(2,1,0,1)', 'k(2,1,0,2)', 'k(2,1,1,0)', 'k(2,1,1,1)',
            'k(2,1,1,2)', 'k(2,1,2,0)', 'k(2,1,2,1)', 'k(2,1,2,2)',
            'k(2,2,0,0)', 'k(2,2,0,1)', 'k(2,2,0,2)', 'k(2,2,1,0)',
            'k(2,2,1,1)', 'k(2,2,1,2)', 'k(2,2,2,0)', 'k(2,2,2,1)',
            'k(2,2,2,2)'
        ]
        self.energy_cutoffs = ['1.8', '2.0', '2.5', '3.0', '4.0']
        self.two_point_k_strings = {
            '1': three_by_three_list,
            '2': ['k(0,0,0,0)'],
            '3': three_by_three_list
        }

        self.matrix_element_properties = {
            '1': {'matrix_element_size': {
                    '1.8': {'k(1,2,1,2)': (9, 4), 'k(1,2,2,1)': (9, 4),
                            'k(2,1,1,2)': (9, 4), 'k(2,1,2,1)': (9, 4)},
                    '2.0': {'k(1,2,1,2)': (9, 4), 'k(1,2,2,1)': (9, 4),
                            'k(2,1,1,2)': (9, 4), 'k(2,1,2,1)': (9, 4)},
                    '2.5': {'k(1,2,1,2)': (9, 4), 'k(1,2,2,1)': (9, 4),
                            'k(2,1,1,2)': (9, 4), 'k(2,1,2,1)': (9, 4)},
                    '3.0': {'k(0,0,0,0)': (9, 4), 'k(0,0,1,2)': (9, 4),
                            'k(0,0,2,1)': (9, 4), 'k(1,2,0,0)': (9, 4),
                            'k(1,2,1,2)': (9, 4), 'k(1,2,2,1)': (9, 4),
                            'k(2,1,0,0)': (9, 4), 'k(2,1,1,2)': (9, 4),
                            'k(2,1,2,1)': (9, 4)},
                    '4.0': {key: (16, 4)
                            for key in self.two_point_k_strings['1']}
            }},
            '2': {'matrix_element_size': {
                    '1.8': {'k(0,0,0,0)': (25, 36)},
                    '2.0': {'k(0,0,0,0)': (25, 36)},
                    '2.5': {'k(0,0,0,0)': (841, 36)},
                    '3.0': {'k(0,0,0,0)': (1089, 324)},
                    '4.0': {'k(0,0,0,0)': (1296, 324)}
            }},
            '3': {'matrix_element_size': {
                    '1.8': {'k(0,0,0,0)': (25, 36)},
                    '2.0': {'k(0,0,0,0)': (25, 36)},
                    '2.5': {'k(0,0,0,0)': (841, 36), 'k(0,0,0,1)': (493, 36),
                            'k(0,0,0,2)': (493, 36), 'k(0,0,1,0)': (493, 36),
                            'k(0,0,1,1)': (493, 36), 'k(0,0,1,2)': (116, 36),
                            'k(0,0,2,0)': (493, 36), 'k(0,0,2,1)': (116, 36),
                            'k(0,0,2,2)': (493, 36), 'k(0,1,0,0)': (493, 36),
                            'k(0,1,0,1)': (289, 36), 'k(0,1,0,2)': (289, 36),
                            'k(0,1,1,0)': (289, 36), 'k(0,1,1,1)': (289, 36),
                            'k(0,1,1,2)': (68, 36), 'k(0,1,2,0)': (289, 36),
                            'k(0,1,2,1)': (68, 36), 'k(0,1,2,2)': (289, 36),
                            'k(0,2,0,0)': (493, 36), 'k(0,2,0,1)': (289, 36),
                            'k(0,2,0,2)': (289, 36), 'k(0,2,1,0)': (289, 36),
                            'k(0,2,1,1)': (289, 36), 'k(0,2,1,2)': (68, 36),
                            'k(0,2,2,0)': (289, 36), 'k(0,2,2,1)': (68, 36),
                            'k(0,2,2,2)': (289, 36), 'k(1,0,0,0)': (493, 36),
                            'k(1,0,0,1)': (289, 36), 'k(1,0,0,2)': (289, 36),
                            'k(1,0,1,0)': (289, 36), 'k(1,0,1,1)': (289, 36),
                            'k(1,0,1,2)': (68, 36), 'k(1,0,2,0)': (289, 36),
                            'k(1,0,2,1)': (68, 36), 'k(1,0,2,2)': (289, 36),
                            'k(1,1,0,0)': (493, 36), 'k(1,1,0,1)': (289, 36),
                            'k(1,1,0,2)': (289, 36), 'k(1,1,1,0)': (289, 36),
                            'k(1,1,1,1)': (289, 36), 'k(1,1,1,2)': (68, 36),
                            'k(1,1,2,0)': (289, 36), 'k(1,1,2,1)': (68, 36),
                            'k(1,1,2,2)': (289, 36), 'k(1,2,0,0)': (116, 36),
                            'k(1,2,0,1)': (68, 36), 'k(1,2,0,2)': (68, 36),
                            'k(1,2,1,0)': (68, 36), 'k(1,2,1,1)': (68, 36),
                            'k(1,2,1,2)': (16, 36), 'k(1,2,2,0)': (68, 36),
                            'k(1,2,2,1)': (16, 36), 'k(1,2,2,2)': (68, 36),
                            'k(2,0,0,0)': (493, 36), 'k(2,0,0,1)': (289, 36),
                            'k(2,0,0,2)': (289, 36), 'k(2,0,1,0)': (289, 36),
                            'k(2,0,1,1)': (289, 36), 'k(2,0,1,2)': (68, 36),
                            'k(2,0,2,0)': (289, 36), 'k(2,0,2,1)': (68, 36),
                            'k(2,0,2,2)': (289, 36), 'k(2,1,0,0)': (116, 36),
                            'k(2,1,0,1)': (68, 36), 'k(2,1,0,2)': (68, 36),
                            'k(2,1,1,0)': (68, 36), 'k(2,1,1,1)': (68, 36),
                            'k(2,1,1,2)': (16, 36), 'k(2,1,2,0)': (68, 36),
                            'k(2,1,2,1)': (16, 36), 'k(2,1,2,2)': (68, 36),
                            'k(2,2,0,0)': (493, 36), 'k(2,2,0,1)': (289, 36),
                            'k(2,2,0,2)': (289, 36), 'k(2,2,1,0)': (289, 36),
                            'k(2,2,1,1)': (289, 36), 'k(2,2,1,2)': (68, 36),
                            'k(2,2,2,0)': (289, 36), 'k(2,2,2,1)': (68, 36),
                            'k(2,2,2,2)': (289, 36)},
                    '3.0': {
                        'k(0,0,0,0)': (1089, 324), 'k(0,0,0,1)': (1089, 324),
                        'k(0,0,0,2)': (1089, 324), 'k(0,0,1,0)': (1089, 324),
                        'k(0,0,1,1)': (1089, 324), 'k(0,0,1,2)': (825, 324),
                        'k(0,0,2,0)': (1089, 324), 'k(0,0,2,1)': (825, 324),
                        'k(0,0,2,2)': (1089, 324), 'k(0,1,0,0)': (1089, 324),
                        'k(0,1,0,1)': (1089, 324), 'k(0,1,0,2)': (1089, 324),
                        'k(0,1,1,0)': (1089, 324), 'k(0,1,1,1)': (1089, 324),
                        'k(0,1,1,2)': (825, 324), 'k(0,1,2,0)': (1089, 324),
                        'k(0,1,2,1)': (825, 324), 'k(0,1,2,2)': (1089, 324),
                        'k(0,2,0,0)': (1089, 324), 'k(0,2,0,1)': (1089, 324),
                        'k(0,2,0,2)': (1089, 324), 'k(0,2,1,0)': (1089, 324),
                        'k(0,2,1,1)': (1089, 324), 'k(0,2,1,2)': (825, 324),
                        'k(0,2,2,0)': (1089, 324), 'k(0,2,2,1)': (825, 324),
                        'k(0,2,2,2)': (1089, 324), 'k(1,0,0,0)': (1089, 324),
                        'k(1,0,0,1)': (1089, 324), 'k(1,0,0,2)': (1089, 324),
                        'k(1,0,1,0)': (1089, 324), 'k(1,0,1,1)': (1089, 324),
                        'k(1,0,1,2)': (825, 324), 'k(1,0,2,0)': (1089, 324),
                        'k(1,0,2,1)': (825, 324), 'k(1,0,2,2)': (1089, 324),
                        'k(1,1,0,0)': (1089, 324), 'k(1,1,0,1)': (1089, 324),
                        'k(1,1,0,2)': (1089, 324), 'k(1,1,1,0)': (1089, 324),
                        'k(1,1,1,1)': (1089, 324), 'k(1,1,1,2)': (825, 324),
                        'k(1,1,2,0)': (1089, 324), 'k(1,1,2,1)': (825, 324),
                        'k(1,1,2,2)': (1089, 324), 'k(1,2,0,0)': (825, 324),
                        'k(1,2,0,1)': (825, 324), 'k(1,2,0,2)': (825, 324),
                        'k(1,2,1,0)': (825, 324), 'k(1,2,1,1)': (825, 324),
                        'k(1,2,1,2)': (625, 324), 'k(1,2,2,0)': (825, 324),
                        'k(1,2,2,1)': (625, 324), 'k(1,2,2,2)': (825, 324),
                        'k(2,0,0,0)': (1089, 324), 'k(2,0,0,1)': (1089, 324),
                        'k(2,0,0,2)': (1089, 324), 'k(2,0,1,0)': (1089, 324),
                        'k(2,0,1,1)': (1089, 324), 'k(2,0,1,2)': (825, 324),
                        'k(2,0,2,0)': (1089, 324), 'k(2,0,2,1)': (825, 324),
                        'k(2,0,2,2)': (1089, 324), 'k(2,1,0,0)': (825, 324),
                        'k(2,1,0,1)': (825, 324), 'k(2,1,0,2)': (825, 324),
                        'k(2,1,1,0)': (825, 324), 'k(2,1,1,1)': (825, 324),
                        'k(2,1,1,2)': (625, 324), 'k(2,1,2,0)': (825, 324),
                        'k(2,1,2,1)': (625, 324), 'k(2,1,2,2)': (825, 324),
                        'k(2,2,0,0)': (1089, 324), 'k(2,2,0,1)': (1089, 324),
                        'k(2,2,0,2)': (1089, 324), 'k(2,2,1,0)': (1089, 324),
                        'k(2,2,1,1)': (1089, 324), 'k(2,2,1,2)': (825, 324),
                        'k(2,2,2,0)': (1089, 324), 'k(2,2,2,1)': (825, 324),
                        'k(2,2,2,2)': (1089, 324)
                    },
                    '4.0': {key: (1296, 324)
                            for key in self.two_point_k_strings['3']}
            }}
        }
        self.vector_modulo_tests = {
            '1': {'v1': {'vector': np.array([0.3, 4.4]),
                         'modulo_vector': np.array([3.49, -1.12524208])},
                  'v2': {'vector': np.array([4.3, -0.3]),
                         'modulo_vector': np.array([1.11, -0.3])},
                  'v3': {'vector': np.array([6, -3.5]),
                         'modulo_vector': np.array([1.215, -0.73737896])}
                  },
            '2': {'v1': {'vector': np.array([6, -3.5]),
                         'modulo_vector': np.array([6, -3.5])},
                  'v2': {'vector': np.array([10, 3.5]),
                         'modulo_vector': np.array([5.215, -4.78786311])},
                  'v3': {'vector': np.array([-10, 20.5]),
                         'modulo_vector': np.array([4.355, -4.36358934])}
                  },
            '3': {'v1': {'vector': np.array([6, -3.5]),
                         'modulo_vector': np.array([6, -3.5])},
                  'v2': {'vector': np.array([10, 3.5]),
                         'modulo_vector': np.array([5.215, -4.78786311])},
                  'v3': {'vector': np.array([-10, 20.5]),
                         'modulo_vector': np.array([4.355, -4.36358934])}
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

    def test_creation_matrix_element_storage(self):
        for test_tag, test_dict in self.matrix_element_properties.items():
            el_shape_dict = test_dict[self.element_key]
            test_file = self.test_input_template % test_tag
            test_path = os.path.join(self.test_input_folder, test_file)
            extb = ExcitonTB(test_path)
            for cutoff_str in self.energy_cutoffs:
                matrix_elem_name = self.element_name_template.format(
                    test_tag,
                    cutoff_str
                )
                extb.create_matrix_element_hdf5(matrix_elem_name,
                                                float(cutoff_str))
                full_path = os.path.join(self.element_dir, matrix_elem_name)
                with File(full_path, 'r') as f:
                    for k_str in self.two_point_k_strings[test_tag]:
                        el_shape = np.array(f['s0'][k_str]['mat_elems']).shape
                        k_str_in_val = bool(
                            k_str in el_shape_dict[cutoff_str]
                        )
                        has_non_zero_shape = bool(np.product(el_shape) > 0)
                        self.assertEqual(k_str_in_val, has_non_zero_shape)

                        if has_non_zero_shape:
                            val_shape = el_shape_dict[cutoff_str][k_str]
                            self.assertTupleEqual(el_shape, val_shape)
                os.remove(full_path)
            extb.terminate_storage_usage()

    def test_vector_modulo_cell(self):
        for test_tag, test_dict in self.vector_modulo_tests.items():
            test_file = self.test_input_template % test_tag
            test_path = os.path.join(self.test_input_folder, test_file)
            extb = ExcitonTB(test_path)
            for vector_str, vector_dict in test_dict.items():
                test_vector = extb.get_vector_modulo_cell(
                    vector=vector_dict['vector'],
                    macro_cell=False
                )
                val_vector = vector_dict['modulo_vector']
                for test_elem, val_elem in zip(test_vector, val_vector):
                    self.assertAlmostEqual(test_elem, val_elem, self.dp)

    def test_fourier_matrix_creation(self):
        for test_tag, test_dict in self.vector_modulo_tests.items():
            test_file = self.test_input_template % test_tag
            test_path = os.path.join(self.test_input_folder, test_file)
            extb = ExcitonTB(test_path)
            exp_size = (extb.n_atoms, extb.n_atoms)
            for kpt in extb.k_grid:
                for idx, pos in enumerate(extb.motif_vectors):
                    fourier_matrix = extb.create_fourier_matrix(pos=pos,
                                                                pos_idx=idx,
                                                                kdiff=kpt)
                    fourier_matrix_c = extb.create_fourier_matrix(pos=pos,
                                                                  pos_idx=idx,
                                                                  kdiff=-kpt)
                    self.assertTupleEqual(fourier_matrix.shape, exp_size)
                    self.assertTupleEqual(fourier_matrix_c.shape, exp_size)
                    self.assertListAlmostEqual(
                        list(fourier_matrix_c.ravel()),
                        list(np.conj(fourier_matrix.ravel())),
                        places=self.dp
                    )


if __name__ == '__main__':
    unittest.main()
