import numpy as np
import matplotlib, sys, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E262, E261

filename = sys.argv[1]

data = np.load(filename)
results = np.median(data['results'], axis=-1)
best_cv = np.argmax(results, axis=-1)
fontnames = data['fontnames']
fontsizes = data['fontsizes']

init_acc = results[:, :, 0] * 19
med_acc = results[:, :, 1] * 21
fin_acc = results[:, :, 2] * 28
accs = [init_acc, med_acc, fin_acc]
maxs = [19, 21, 28]


# Fontsize vs Accuracy Plot
f, axes = plt.subplots(3, figsize=(3, 7))
for ii, font in enumerate(fontnames):
    nums = np.arange(len(fontsizes))
    for jj in range(3):
        idxs = best_cv[ii, :, jj]
        axes[jj].plot(fontsizes, accs[jj][ii][nums, idxs],
                      label=os.path.splitext(font)[0])
for ii, ax in enumerate(axes):
    ax.set_xlabel('Font Sizes')
    ax.set_ylabel('Accuracy/chance')
    ax.axhline(maxs[ii], linestyle='--', c='black')
axes[0].set_title('Initial')
axes[1].set_title('Medial')
axes[2].set_title('Final')
f.tight_layout()
plt.savefig(os.path.join(os.environ['HOME'],
                         'results/hangul/single_logreg_accuracy.pdf'))
plt.savefig(os.path.join(os.environ['HOME'],
                         'results/hangul/single_logreg_accuracy.png'))
