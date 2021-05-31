import argparse

from hfd.dataset_creation import txt2png


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Create pngs for a font, ' +
                                                  'fontsize pair.'))
    parser.add_argument('base_path', type=str,
                        help='path to hangul folder')
    parser.add_argument('font_file', type=str,
                        help='font filename (ttf, otf)')
    parser.add_argument('fontsize', type=int, nargs='+',
                        help='fontsize(s) for pngs')

    args = parser.parse_args()

    txt2png(args.base_path, args.font_file, args.fontsize)
