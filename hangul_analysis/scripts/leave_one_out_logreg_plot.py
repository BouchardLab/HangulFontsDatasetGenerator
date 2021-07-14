import numpy as np
import matplotlib, sys, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E262, E261

filename = sys.argv[1]

data = np.load(filename)
results = data['results']
best_cv = np.argmax(results, axis=1)
fontnames = data['fontnames']
fontsize = data['fontsize']

maxs = np.array([19, 21, 28])


# Fontsize vs Accuracy Plot
f, ax = plt.subplots(1, figsize=(3, 7. / 3))
print(results.shape)
for ii, font in enumerate(fontnames):
    nums = np.arange(3)
    idxs = best_cv[ii]
    ax.scatter(nums, results[ii][idxs, nums] * maxs,
               marker='.', label=os.path.splitext(font)[0])
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Initial', 'Medial', 'Final'])
ax.set_ylabel('Accuracy/chance')
for ii in range(3):
    ax.plot([ii - .3, ii + .3], [maxs[ii], maxs[ii]],
            linestyle='--', c='black')
ax.set_ylim(1, 30)
f.tight_layout()
plt.savefig(os.path.join(os.environ['HOME'],
                         'results/hangul/leave_one_out_logreg_accuracy.pdf'))
plt.savefig(os.path.join(os.environ['HOME'],
                         'results/hangul/leave_one_out_logreg_accuracy.png'))
