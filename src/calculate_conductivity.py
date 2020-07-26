import sys
import argparse

from exciton_tb.exciton_tb import ExcitonTB
from conductivity.conductivity_tb import ConductivityTB
# from util.writers import write_conductivity


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_hdf5', type=str)
    parser.add_argument('-o', '--output-name', default=None, type=str)
    parser.add_argument('-t', '--terminal-output', action='store_true')
    parser.add_argument('-vm', '--velocity-matrix', default=None)
    parser.add_argument('-rpa', '--use-rpa', action='store_true')
    parser.add_argument('-i', '--interaction', default='keldysh', type=str)
    parser.add_argument('-w', '--broadening-width', default=0.01, type=float)
    parser.add_argument('-fmin', '--frequency-min', default=0.0, type=float)
    parser.add_argument('-fmax', '--frequency-max', default=6.0, type=float)
    parser.add_argument('-fnum', '--frequency-num', default=1000, type=int)
    parser.add_argument('-d', '--dielectric', action='store_true')
    parser.add_argument('-b', '--broadening', defaurlt='lorentz', type=str)
    parser.add_argument('-mn', '--matrix-element', default='temp_elem.hdf5')
    return parser

def main():
    parser = get_argparser()
    args = parser.parse_args()

    if args.output_name is None and (not args.terminal_output):
        raise Exception("Need to either specify output file (-o <output name>)\
                         or use terminal output (-t)!")

    freq_range = (args.frequency_min, args.frequency_max, args.frequency_num)

    exc_tb = ExcitonTB(args.input_hdf5)

    if not args.use_rpa:
        exc_tb.create_matrix_element_hdf5(args.matrix_element)

    cond_tb = ConductivityTB(exciton_obj=exc_tb)
    frequencies, conductivity = cond_tb.interacting_conductivity(
        sigma=args.broadening_width,
        freq_range=freq_range,
        imag_dielectric=args.dielectric,
        broadening=args.broadening
    )

    exc_tb.terminate_storage_usage()

    if args.terminal_output:
        write_obj = sys.stdout
        write_conductivity(frequencies=frequencies,
                           conductivity=conductivity,
                           write_obj=write_obj)
    else:
        with open(args.output_name, 'w') as write_obj:
            write_conductivity(frequencies=frequencies,
                               conductivity=conductivity,
                               write_obj=write_obj)

if __name__ == '__main__':
    main()
