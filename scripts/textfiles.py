"""Write all hangul characters to text files
Text files are already provided in the github repo, but if you want to
regenerate them, this will do it.
"""
import sys, os
from builtins import chr

parser = argparse.ArgumentParser(description='Generate hangul text files.')
parser.add_argument('folder', type=str, help='Folder to save the folder files inside.')

parser.parse_args()
folder = args.folder
folder = os.path.join(folder, 'texts')

starts = [44032, 4352, 4449, 4520]
ends = [55204, 4371, 4470, 4547]

try:
    os.mkdir(folder)
except FileExistsError:
    pass

for start, end in zip(starts, ends):
    for num in range(start, end):
        char = chr(num)
        h = hex(num)
        name = h[2:].upper()
        with open(os.path.join(folder, name + '.txt'), 'w') as f:
            f.write(char)
