# Save a folder of hangul png to a hdf5 file

import argparse

from hangul.dataset_creation import png2h5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create h5 file from pngs.')
    parser.add_argument('base_folder', type=str,
                        help='path to hangul folder')
    parser.add_argument('font_name', type=str,
                        help='font name')
    parser.add_argument('fontsize', type=int,
                        help='fontsize for pngs')

    args = parser.parse_args()

    png2h5(args.base_folder, args.font_name, args.fontsize)
