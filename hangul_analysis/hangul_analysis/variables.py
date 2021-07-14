"""Generative variable labels for different labellings of the blocks.
The 4 types of labels are
- initial, medial, and final labels (imf_labels)
- imf geometry labels (imf_style_labels), and
- atom glyph labels not taking rotations into account (atom_labels)
- atom glyph labels taking rotations into account (atom_mod_rotations_labels)

All variables are contined in the label_df object.
"""

import numpy as np

import pandas as pd

from itertools import product


label_df = pd.DataFrame()

# imf variables
n_initial = 19
n_medial = 21
n_final = 28
n_blocks = n_initial * n_medial * n_final

initial = np.arange(28 * 18, (28 * 18) + (588 * 19), 588, dtype=int)
medial = np.arange(588 * 11, (588 * 11) + (28 * 21), 28, dtype=int)
final = np.arange((588 * 11), (588 * 11) + 28, 1, dtype=int)

imf_labels = np.zeros((n_blocks, 3), dtype=int)
for idx, (ii, jj, kk) in enumerate(product(range(n_initial),
                                           range(n_medial),
                                           range(n_final))):
    imf_labels[idx] = [ii, jj, kk]
label_df['initial'] = imf_labels[:, 0]
label_df['medial'] = imf_labels[:, 1]
label_df['final'] = imf_labels[:, 2]


# geometric imf style variables
initial_style = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=int)
medial_style = np.array([0, 1, 0, 1, 0, 1, 0, 1,
                         2, 3, 4, 3, 2, 2, 3, 4, 3, 2, 2, 3, 0, ], dtype=int)
final_style = np.array([0, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1], dtype=int)
style_arrays = [initial_style, medial_style, final_style]

imf_style_labels =  np.zeros_like(imf_labels)
for ii in range(3):
    imf_style_labels[:, ii] = style_arrays[ii][imf_labels[:, ii]]
new_labels = np.zeros((2*5*3, 3), dtype=int)
for nn, (ii, jj, kk) in enumerate(product(range(2), range(5), range(3))):
    new_labels[nn, 0] = ii
    new_labels[nn, 1] = jj
    new_labels[nn, 2] = kk
y_all = np.zeros(n_blocks, dtype=int)
for ii, r in enumerate(imf_style_labels):
    y_all[ii] = np.where(np.prod(r[np.newaxis] == new_labels, axis=1))[0]

label_df['initial_geometry'] = imf_style_labels[:, 0]
label_df['medial_geometry'] = imf_style_labels[:, 1]
label_df['final_geometry'] = imf_style_labels[:, 2]
label_df['all_geometry'] = y_all


# Medial mod rotations (no composition)
medial_glyph_atom= np.array([0, 1, 0, 1, 0, 1, 0, 1,
                         2, 3, 4, 3, 2, 2, 3, 4, 3, 2, 2, 3, 0, ], dtype=int)


# Atomic glyph variables
# For each atomic glyph mod rotations, where does it appear (first)
atoms_mod_rotations = np.zeros([16, 3], dtype=int)
atom_idx = 0
for initial_atom in [0, 3, 5, 6, 7, 9, 11, 12, 14, 15, 16, 17, 18]:
    atoms_mod_rotations[atom_idx] = -1
    atoms_mod_rotations[atom_idx, 0] = initial_atom
    atom_idx += 1
for medial_atom in [0, 2, 18]:
    atoms_mod_rotations[atom_idx] = -1
    atoms_mod_rotations[atom_idx, 1] = medial_atom
    atom_idx += 1

# For each atomic glyph, where does it appear (first)
atoms = np.zeros([24, 3], dtype=int)
atom_idx = 0
for initial_atom in [0, 2, 3, 5, 6, 7, 9, 11, 12, 14, 15, 16, 17, 18]:
    atoms[atom_idx] = -1
    atoms[atom_idx, 0] = initial_atom
    atom_idx += 1
for medial_atom in [0, 2, 4, 6, 8, 12, 13, 17, 18, 20]:
    atoms[atom_idx] = -1
    atoms[atom_idx, 1] = medial_atom
    atom_idx += 1

"""
atoms to atoms_mod_rotations glyph correspondence.
These are indices, not labels. A labels of 0 corresponds to no glyph.
"""
atom_correspondence = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                13, 14, 13, 14, 13, 14, 13, 14, 15, 15])

# 0 indicates no glyph
initial_atoms = np.array([[1, 0],
                          [1, 1],
                          [2, 0],
                          [3, 0],
                          [3, 3],
                          [4, 0],
                          [5, 0],
                          [6, 0],
                          [6, 6],
                          [7, 0],
                          [7, 7],
                          [8, 0],
                          [9, 0],
                          [9, 9],
                          [10, 0],
                          [11, 0],
                          [12, 0],
                          [13, 0],
                          [14, 0]], dtype=int)
medial_atoms = np.array([[0, 15, 0],
                         [0, 15, 24 ],
                         [0, 16, 0 ],
                         [0, 16, 24],
                         [0, 17, 0],
                         [0, 17, 24],
                         [0, 18, 0],
                         [0, 18, 24],
                         [19, 0, 0],
                         [19, 15, 0],
                         [19, 15, 24],
                         [19, 24, 0],
                         [20, 0, 0],
                         [21, 0, 0],
                         [21, 17, 0],
                         [21, 17, 24],
                         [21, 24, 0],
                         [22, 0, 0],
                         [23, 0, 0],
                         [23, 24, 0],
                         [0, 24, 0]], dtype=int)
final_atoms = np.array([[0, 0],
                        [1, 0],
                        [1, 1],
                        [1, 7],
                        [2, 0],
                        [2, 9],
                        [2, 14],
                        [3, 0],
                        [4, 0],
                        [4, 1],
                        [4, 5],
                        [4, 6],
                        [4, 7],
                        [4, 12],
                        [4, 13],
                        [4, 14],
                        [5, 0],
                        [6, 0],
                        [6, 7],
                        [7, 0],
                        [7, 7],
                        [8, 0],
                        [9, 0],
                        [10, 0],
                        [11, 0],
                        [12, 0],
                        [13, 0],
                        [14, 0]])

def idx_to_label(idx):
    if idx == 0:
        return 0
    else:
        return atom_correspondence[idx - 1] + 1
initial_atom_labels = np.zeros((n_blocks, initial_atoms.shape[1]), dtype=int)
medial_atom_labels = np.zeros((n_blocks, medial_atoms.shape[1]), dtype=int)
final_atom_labels = np.zeros((n_blocks, final_atoms.shape[1]), dtype=int)
initial_atom_mod_rotations_labels = np.zeros_like(initial_atom_labels)
medial_atom_mod_rotations_labels = np.zeros_like(medial_atom_labels)
final_atom_mod_rotations_labels = np.zeros_like(final_atom_labels)
for ii in range(n_blocks):
    init_idx, med_idx, fin_idx = imf_labels[ii]
    initial_atom_labels[ii] = initial_atoms[init_idx]
    medial_atom_labels[ii] = medial_atoms[med_idx]
    final_atom_labels[ii] = final_atoms[fin_idx]
    initial_atom_mod_rotations_labels[ii] = [idx_to_label(idx) for idx in initial_atoms[init_idx]]
    medial_atom_mod_rotations_labels[ii] = [idx_to_label(idx) for idx in medial_atoms[med_idx]]
    final_atom_mod_rotations_labels[ii] = [idx_to_label(idx) for idx in final_atoms[fin_idx]]

atom_labels = [initial_atom_labels, medial_atom_labels, final_atom_labels]
atom_mod_rotations_labels = [initial_atom_mod_rotations_labels,
                             medial_atom_mod_rotations_labels,
                             final_atom_mod_rotations_labels]
for ii in range(2):
    label_df[('initial_atom', ii)] = atom_labels[0][:, ii]
    label_df[('initial_atom_mod_rotations', ii)] = atom_mod_rotations_labels[0][:, ii]
for ii in range(3):
    label_df[('medial_atom', ii)] = atom_labels[1][:, ii]
    label_df[('medial_atom_mod_rotations', ii)] = atom_mod_rotations_labels[1][:, ii]
for ii in range(2):
    label_df[('final_atom', ii)] = atom_labels[2][:, ii]
    label_df[('final_atom_mod_rotations', ii)] = atom_mod_rotations_labels[2][:, ii]

atom_bof = np.zeros((n_blocks, atoms.shape[0]), dtype=int)
atom_mod_rotations_bof = np.zeros((n_blocks, atoms_mod_rotations.shape[0]), dtype=int)
for ii in range(n_blocks):
    init_idx, med_idx, fin_idx = imf_labels[ii]
    for idx in initial_atoms[init_idx]:
        if idx > 0:
            atom_bof[ii, idx-1] = 1
            atom_mod_rotations_bof[ii, idx_to_label(idx)-1] = 1
    for idx in medial_atoms[init_idx]:
        if idx > 0:
            atom_bof[ii, idx-1] = 1
            atom_mod_rotations_bof[ii, idx_to_label(idx)-1] = 1
    for idx in final_atoms[init_idx]:
        if idx > 0:
            atom_bof[ii, idx-1] = 1
            atom_mod_rotations_bof[ii, idx_to_label(idx)-1] = 1

label_df['atom_bof'] = pd.Series(data=[ai for ai in atom_bof])
label_df['atom_mod_rotations_bof'] = pd.Series(data=[ai for ai in atom_mod_rotations_bof])
