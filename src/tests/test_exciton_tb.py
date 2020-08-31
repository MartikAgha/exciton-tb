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
    element_name_template_1 = 'matrix_element_1_{}_{}.hdf5'
    element_name_template_2 = 'matrix_element_2_{}_{}.hdf5'
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
                    '1.8': {'k(1,2,1,2)': (4, 4), 'k(1,2,2,1)': (4, 4),
                            'k(2,1,1,2)': (4, 4), 'k(2,1,2,1)': (4, 4)},
                    '2.0': {'k(1,2,1,2)': (4, 4), 'k(1,2,2,1)': (4, 4),
                            'k(2,1,1,2)': (4, 4), 'k(2,1,2,1)': (4, 4)},
                    '2.5': {'k(1,2,1,2)': (4, 4), 'k(1,2,2,1)': (4, 4),
                            'k(2,1,1,2)': (4, 4), 'k(2,1,2,1)': (4, 4)},
                    '3.0': {'k(0,0,0,0)': (4, 4), 'k(0,0,1,2)': (4, 4),
                            'k(0,0,2,1)': (4, 4), 'k(1,2,0,0)': (4, 4),
                            'k(1,2,1,2)': (4, 4), 'k(1,2,2,1)': (4, 4),
                            'k(2,1,0,0)': (4, 4), 'k(2,1,1,2)': (4, 4),
                            'k(2,1,2,1)': (4, 4)},
                    '4.0': {key: (16, 4)
                            for key in self.two_point_k_strings['1']}
            }},
            '2': {'matrix_element_size': {
                    '1.8': {'k(0,0,0,0)': (16, 36)},
                    '2.0': {'k(0,0,0,0)': (16, 36)},
                    '2.5': {'k(0,0,0,0)': (784, 36)},
                    '3.0': {'k(0,0,0,0)': (1024, 324)},
                    '4.0': {'k(0,0,0,0)': (1296, 324)}
            }},
            '3': {'matrix_element_size': {
                    '1.8': {'k(0,0,0,0)': (16, 36)},
                    '2.0': {'k(0,0,0,0)': (16, 36)},
                    '2.5': {'k(0,0,0,0)': (784, 36), 'k(0,0,0,1)': (448, 36),
                            'k(0,0,0,2)': (448, 36), 'k(0,0,1,0)': (448, 36),
                            'k(0,0,1,1)': (448, 36), 'k(0,0,1,2)': (84, 36),
                            'k(0,0,2,0)': (448, 36), 'k(0,0,2,1)': (84, 36),
                            'k(0,0,2,2)': (448, 36), 'k(0,1,0,0)': (448, 36),
                            'k(0,1,0,1)': (256, 36), 'k(0,1,0,2)': (256, 36),
                            'k(0,1,1,0)': (256, 36), 'k(0,1,1,1)': (256, 36),
                            'k(0,1,1,2)': (48, 36), 'k(0,1,2,0)': (256, 36),
                            'k(0,1,2,1)': (48, 36), 'k(0,1,2,2)': (256, 36),
                            'k(0,2,0,0)': (448, 36), 'k(0,2,0,1)': (256, 36),
                            'k(0,2,0,2)': (256, 36), 'k(0,2,1,0)': (256, 36),
                            'k(0,2,1,1)': (256, 36), 'k(0,2,1,2)': (48, 36),
                            'k(0,2,2,0)': (256, 36), 'k(0,2,2,1)': (48, 36),
                            'k(0,2,2,2)': (256, 36), 'k(1,0,0,0)': (448, 36),
                            'k(1,0,0,1)': (256, 36), 'k(1,0,0,2)': (256, 36),
                            'k(1,0,1,0)': (256, 36), 'k(1,0,1,1)': (256, 36),
                            'k(1,0,1,2)': (48, 36), 'k(1,0,2,0)': (256, 36),
                            'k(1,0,2,1)': (48, 36), 'k(1,0,2,2)': (256, 36),
                            'k(1,1,0,0)': (448, 36), 'k(1,1,0,1)': (256, 36),
                            'k(1,1,0,2)': (256, 36), 'k(1,1,1,0)': (256, 36),
                            'k(1,1,1,1)': (256, 36), 'k(1,1,1,2)': (48, 36),
                            'k(1,1,2,0)': (256, 36), 'k(1,1,2,1)': (48, 36),
                            'k(1,1,2,2)': (256, 36), 'k(1,2,0,0)': (84, 36),
                            'k(1,2,0,1)': (48, 36), 'k(1,2,0,2)': (48, 36),
                            'k(1,2,1,0)': (48, 36), 'k(1,2,1,1)': (48, 36),
                            'k(1,2,1,2)': (9, 36), 'k(1,2,2,0)': (48, 36),
                            'k(1,2,2,1)': (9, 36), 'k(1,2,2,2)': (48, 36),
                            'k(2,0,0,0)': (448, 36), 'k(2,0,0,1)': (256, 36),
                            'k(2,0,0,2)': (256, 36), 'k(2,0,1,0)': (256, 36),
                            'k(2,0,1,1)': (256, 36), 'k(2,0,1,2)': (48, 36),
                            'k(2,0,2,0)': (256, 36), 'k(2,0,2,1)': (48, 36),
                            'k(2,0,2,2)': (256, 36), 'k(2,1,0,0)': (84, 36),
                            'k(2,1,0,1)': (48, 36), 'k(2,1,0,2)': (48, 36),
                            'k(2,1,1,0)': (48, 36), 'k(2,1,1,1)': (48, 36),
                            'k(2,1,1,2)': (9, 36), 'k(2,1,2,0)': (48, 36),
                            'k(2,1,2,1)': (9, 36), 'k(2,1,2,2)': (48, 36),
                            'k(2,2,0,0)': (448, 36), 'k(2,2,0,1)': (256, 36),
                            'k(2,2,0,2)': (256, 36), 'k(2,2,1,0)': (256, 36),
                            'k(2,2,1,1)': (256, 36), 'k(2,2,1,2)': (48, 36),
                            'k(2,2,2,0)': (256, 36), 'k(2,2,2,1)': (48, 36),
                            'k(2,2,2,2)': (256, 36)},
                    '3.0': {
                        'k(0,0,0,0)': (1024, 324), 'k(0,0,0,1)': (1024, 324),
                        'k(0,0,0,2)': (1024, 324), 'k(0,0,1,0)': (1024, 324),
                        'k(0,0,1,1)': (1024, 324), 'k(0,0,1,2)': (768, 324),
                        'k(0,0,2,0)': (1024, 324), 'k(0,0,2,1)': (768, 324),
                        'k(0,0,2,2)': (1024, 324), 'k(0,1,0,0)': (1024, 324),
                        'k(0,1,0,1)': (1024, 324), 'k(0,1,0,2)': (1024, 324),
                        'k(0,1,1,0)': (1024, 324), 'k(0,1,1,1)': (1024, 324),
                        'k(0,1,1,2)': (768, 324), 'k(0,1,2,0)': (1024, 324),
                        'k(0,1,2,1)': (768, 324), 'k(0,1,2,2)': (1024, 324),
                        'k(0,2,0,0)': (1024, 324), 'k(0,2,0,1)': (1024, 324),
                        'k(0,2,0,2)': (1024, 324), 'k(0,2,1,0)': (1024, 324),
                        'k(0,2,1,1)': (1024, 324), 'k(0,2,1,2)': (768, 324),
                        'k(0,2,2,0)': (1024, 324), 'k(0,2,2,1)': (768, 324),
                        'k(0,2,2,2)': (1024, 324), 'k(1,0,0,0)': (1024, 324),
                        'k(1,0,0,1)': (1024, 324), 'k(1,0,0,2)': (1024, 324),
                        'k(1,0,1,0)': (1024, 324), 'k(1,0,1,1)': (1024, 324),
                        'k(1,0,1,2)': (768, 324), 'k(1,0,2,0)': (1024, 324),
                        'k(1,0,2,1)': (768, 324), 'k(1,0,2,2)': (1024, 324),
                        'k(1,1,0,0)': (1024, 324), 'k(1,1,0,1)': (1024, 324),
                        'k(1,1,0,2)': (1024, 324), 'k(1,1,1,0)': (1024, 324),
                        'k(1,1,1,1)': (1024, 324), 'k(1,1,1,2)': (768, 324),
                        'k(1,1,2,0)': (1024, 324), 'k(1,1,2,1)': (768, 324),
                        'k(1,1,2,2)': (1024, 324), 'k(1,2,0,0)': (768, 324),
                        'k(1,2,0,1)': (768, 324), 'k(1,2,0,2)': (768, 324),
                        'k(1,2,1,0)': (768, 324), 'k(1,2,1,1)': (768, 324),
                        'k(1,2,1,2)': (576, 324), 'k(1,2,2,0)': (768, 324),
                        'k(1,2,2,1)': (576, 324), 'k(1,2,2,2)': (768, 324),
                        'k(2,0,0,0)': (1024, 324), 'k(2,0,0,1)': (1024, 324),
                        'k(2,0,0,2)': (1024, 324), 'k(2,0,1,0)': (1024, 324),
                        'k(2,0,1,1)': (1024, 324), 'k(2,0,1,2)': (768, 324),
                        'k(2,0,2,0)': (1024, 324), 'k(2,0,2,1)': (768, 324),
                        'k(2,0,2,2)': (1024, 324), 'k(2,1,0,0)': (768, 324),
                        'k(2,1,0,1)': (768, 324), 'k(2,1,0,2)': (768, 324),
                        'k(2,1,1,0)': (768, 324), 'k(2,1,1,1)': (768, 324),
                        'k(2,1,1,2)': (576, 324), 'k(2,1,2,0)': (768, 324),
                        'k(2,1,2,1)': (576, 324), 'k(2,1,2,2)': (768, 324),
                        'k(2,2,0,0)': (1024, 324), 'k(2,2,0,1)': (1024, 324),
                        'k(2,2,0,2)': (1024, 324), 'k(2,2,1,0)': (1024, 324),
                        'k(2,2,1,1)': (1024, 324), 'k(2,2,1,2)': (768, 324),
                        'k(2,2,2,0)': (1024, 324), 'k(2,2,2,1)': (768, 324),
                        'k(2,2,2,2)': (1024, 324)
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
        self.matrix_dim_block_starts_tests = {
            '1': {
                '1.8': {'matrix_dim': 8,
                        'block_start': [0, 0, 0, 0, 0, 0, 4, 4, 8]},
                '2.0': {'matrix_dim': 8,
                        'block_start': [0, 0, 0, 0, 0, 0, 4, 4, 8]},
                '2.5': {'matrix_dim': 8,
                        'block_start': [0, 0, 0, 0, 0, 0, 4, 4, 8]},
                '3.0': {'matrix_dim': 12,
                        'block_start': [0, 4, 4, 4, 4, 4, 8, 8, 12]},
                '4.0': {'matrix_dim': 72,
                        'block_start': [0, 8, 16, 24, 32, 40, 48, 56, 64]},
            },
            '2': {
                '1.8': {'matrix_dim': 24,
                        'block_start': [0]},
                '2.0': {'matrix_dim': 24,
                        'block_start': [0]},
                '2.5': {'matrix_dim': 168,
                        'block_start': [0]},
                '3.0': {'matrix_dim': 576,
                        'block_start': [0]},
                '4.0': {'matrix_dim': 648,
                        'block_start': [0]},
            },
            '3': {
                '1.8': {'matrix_dim': 24,
                        'block_start': [0, 24, 24, 24, 24, 24, 24, 24, 24]},
                '2.0': {'matrix_dim': 24,
                        'block_start': [0, 24, 24, 24, 24, 24, 24, 24, 24]},
                '2.5': {'matrix_dim': 780,
                        'block_start': [0, 168, 264, 360, 456,
                                        552, 570, 666, 684]},
                '3.0': {'matrix_dim': 4896,
                        'block_start': [0, 576, 1152, 1728, 2304,
                                        2880, 3312, 3888, 4320]},
                '4.0': {'matrix_dim': 5832,
                        'block_start': [0, 648, 1296, 1944, 2592,
                                        3240, 3888, 4536, 5184]},
            }
        }

        self.eigenvalue_cutoff_tests = {
            'test_1': {
                'eigenvalues': np.array([-2.0, -1.5, -1.0, -0.5,
                                         0.5, 1.0, 1.5, 2.0]),
                'energy_cutoff': [1.001, 1.501, 2.001, 2.501],
                'edge_index': 4,
                'cb_max': [4, 5, 6, 7],
                'vb_min': [3, 2, 1, 0]
            },
            'test_2': {
                'eigenvalues': np.linspace(-5.0, 5.0, 200),
                'energy_cutoff': [0.5, 1.0, 1.5, 2.0, 2.5],
                'edge_index': 100,
                'cb_max': [108, 118, 128, 138, 148],
                'vb_min': [91, 81, 71, 61, 51]
            },
            'test_3': {
                'eigenvalues': np.logspace(-1, 3, 200),
                'energy_cutoff': [1.0, 2.0, 3.0, 4.0, 5.0],
                'edge_index': 83,
                'cb_max': [86, 90, 93, 95, 98],
                'vb_min': [78, 71, 61, 41, 0]
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
                matrix_elem_name = self.element_name_template_1.format(
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

    def test_matrix_dim_and_block_starts(self):
        for test_tag, test_dict in self.matrix_dim_block_starts_tests.items():
            test_file = self.test_input_template % test_tag
            test_path = os.path.join(self.test_input_folder, test_file)
            extb = ExcitonTB(test_path)
            for cutoff_str in self.energy_cutoffs:
                matrix_elem_name = self.element_name_template_2.format(
                    test_tag,
                    cutoff_str
                )
                extb.create_matrix_element_hdf5(matrix_elem_name,
                                                float(cutoff_str))
                val_dict = test_dict[cutoff_str]

                full_path = os.path.join(self.element_dir, matrix_elem_name)
                with File(full_path, 'r') as f:
                    test_data = extb.get_matrix_dim_and_block_starts(f)
                os.remove(full_path)

                test_mat_size = test_data[0][0]
                val_mat_size = val_dict['matrix_dim']
                self.assertEqual(test_mat_size, val_mat_size)

                test_block_starts = test_data[1][0]
                val_block_starts = val_dict['block_start']
                self.assertListEqual(test_block_starts, val_block_starts)

    def test_energy_cutoff_band_indices(self):
        for tag, example in self.eigenvalue_cutoff_tests.items():

            for idx, cutoff in enumerate(example['energy_cutoff']):
                cb_max, vb_min = ExcitonTB.get_band_extrema(
                    eigenvalues=example['eigenvalues'],
                    energy_cutoff=cutoff,
                    edge_index=example['edge_index']
                )
                self.assertEqual(cb_max, example['cb_max'][idx])
                self.assertEqual(vb_min, example['vb_min'][idx])


if __name__ == '__main__':
    unittest.main()
