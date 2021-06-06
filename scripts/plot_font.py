import argparse, glob, os

from hangul.plotting import plot_single_font


def plot_font(h5_path):
    folder, _ = os.path.split(h5_path)
    base_folder, _ = os.path.split(folder)
    files = glob.glob(h5_path)

    for f in files:
        _, f_name = os.path.split(f)
        name, _ = os.path.splitext(f_name)
        pdf_name = os.path.join(base_folder, 'pdfs', '{}.pdf'.format(name))
        plot_single_font(f, pdf_name)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot all blocks in an pdf file.')
    parser.add_argument('h5_path', type=str,
                        help='path to png folder')

    args = parser.parse_args()
    plot_font(args.h5_path)
