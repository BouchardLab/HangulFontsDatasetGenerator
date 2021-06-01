import argparse, os
from multiprocessing import Pool

from hfd.dataset_creation import _txt2png, _png2h5

from hfd.plotting import _plot_single_font


parser = argparse.ArgumentParser(description='Preprocess all fonts.')
parser.add_argument('hangul_folder', type=str,
                    help='path to base')
parser.add_argument('fontsize', type=int, default=None,
                    help='single fontsize')
parser.add_argument('--fontname', type=str, default=None,
                    help='single fontname')
args = parser.parse_args()

base_folder = args.hangul_folder

if args.fontsize is None:
    fontsizes = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
                 30, 36, 42, 48, 54, 60, 66, 72]
else:
    fontsizes = [int(args.fontsize)]

if args.fontname is None:
    font_files = next(os.walk(os.path.join(base_folder, 'all_fonts')))[-1]
else:
    font_files = [args.fontname]

print(os.path.splitext(font_files[0])[-1].lower())
font_files = [f for f in font_files if
              (os.path.splitext(f)[-1].lower() in ['.ttf', '.otf'])]
font_names = [os.path.splitext(f)[0] for f in font_files]
texts_path = os.path.join(base_folder, 'texts')
print(base_folder)
print(texts_path)
print(font_files)

# txt2png
try:
    os.mkdir(os.path.join(base_folder, 'pngs'))
except FileExistsError:
    pass
txt2png_args = []
for f in font_files:
    font_path = os.path.join(base_folder, 'all_fonts', f)
    for fontsize in fontsizes:
        txt2png_args.append((base_folder, f, fontsize))
with Pool() as p:
    p.map(_txt2png, txt2png_args)

# png2h5
print('Creating h5s')
try:
    os.mkdir(os.path.join(base_folder, 'h5s'))
except FileExistsError:
    pass
image_folders = font_names
png2h5_args = []
for f in image_folders:
    image_path = os.path.join(base_folder, 'pngs', f)
    for fontsize in fontsizes:
        png2h5_args.append((base_folder, f, fontsize))
with Pool() as p:
    p.map(_png2h5, png2h5_args)

# plot pdfs
print('Creating pdfs')
try:
    os.mkdir(os.path.join(base_folder, 'pdfs'))
except FileExistsError:
    pass
h5_folders = font_names
plot_args = []
for f in h5_folders:
    try:
        os.mkdir(os.path.join(base_folder, 'pdfs', f))
    except FileExistsError:
        pass
    h5_path = os.path.join(base_folder, 'h5s', f)
    fontsize = sorted(fontsizes)[-1]
    plot_args.append((base_folder, f, fontsize))
with Pool() as p:
    p.map(_plot_single_font, plot_args)
