import sys
import argparse

from exciton_tb.exciton_tb import ExcitonTB
from exciton_tb.exciton_tools import extract_exciton_dos
from util.writers import write_excitons, write_exciton_dos


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_hdf5', type=str)
    parser.add_argument('-o', '--output-name', default=None, type=str)
    parser.add_argument('-t', '--terminal-output', action='store_true')
    parser.add_argument('-i', '--interaction', default='keldysh', type=str)
    parser.add_argument('-dos', '--exciton-dos', action='store_true')
    parser.add_argument('-fmin', '--frequency-min', default=0.0, type=float)
    parser.add_argument('-fmax', '--frequency-max', default=6.0, type=float)
    parser.add_argument('-fnum', '--frequency-num', default=1000, type=int)
    parser.add_argument('-w', '--broadening-width', default=0.01, type=float)
    parser.add_argument('-b', '--broadening', default='lorentz', type=str)
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
    exc_tb.create_matrix_element_hdf5(storage_name=args.matrix_element)
    output = exc_tb.get_bse_eigensystem_direct(solve=True)
    exc_tb.terminate_storage_usage()
    if args.exciton_dos:
        output = extract_exciton_dos(excitons=output,
                                     frequencies=freq_range,
                                     broadening_function=args.broadening,
                                     sigma=args.broadening_width,
                                     spin_split=bool(exc_tb.n_spins == 2))
    if args.terminal_output:
        write_obj = sys.stdout
        if args.exciton_dos:
            write_exciton_dos(output, write_obj=write_obj)
        else:
            write_excitons(output, write_obj=write_obj)
    else:
        with open(args.output_name, 'w') as write_obj:
            if args.exciton_dos:
                write_exciton_dos(output, write_obj=write_obj)
            else:
                write_excitons(output, write_obj=write_obj)


if __name__ == '__main__':
    main()
