# Write all hangul characters to text files
import sys, os
from builtins import chr


try:
    folder = sys.argv[1]
except IndexError:
    folder = '..'
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
