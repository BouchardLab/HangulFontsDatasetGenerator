import numpy as np

from sklearn.cluster import MiniBatchKMeans
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import ward, cut_tree


def compare_rep_to_label(rep, y, kind='knn'):
    """Compare a representation, rep, to a set of labels, y. Fits a k-means
    model to the representation where k is the number of labels. Finds the
    best match between the clusters and labels and compares them.

    Parameters
    ----------
    rep: ndarray (n_samples, n_features)
        Representation of the data.
    y: ndarray (n_samples, ) int
        Labels for data.

    Returns
    -------
    cluster_idx: ndarray, int (n_classes,)
        Indices of clusters to match with labels.
    label_idx: ndarray, int (n_classes,)
        Indices of labels to match with clusters.
    cost_matrix: ndarray (n_classes, n_classes)
        Cost matrix under original orderings.
    new_cost_matrix: ndarray (n_classes, n_classes)
        Cost matrix under aligned, sorted orderings.
    new_y: ndarray, int (n_classes,)
        Labels under the best cluster labelling.
    accuracy: float
        Labels accuracy under the best cluster labelling.
    null_accuracy: float
        Labels accuracy under a permutation of the best cluster labelling.
    """
    if kind not in ['ward', 'knn', 'miniknn']:
        raise ValueError
    dim = y.max() + 1

    if kind == 'ward':
        Z = ward(rep)
        yk = cut_tree(Z, n_clusters=dim)
    elif kind == 'knn':
        km = MiniBatchKMeans(dim)
        km.fit(rep)
        yk = km.labels_
    else:
        raise ValueError

    cost_matrix = np.zeros((dim, dim))
    for ii in range(dim):
        e1 = yk == ii
        for jj in range(dim):
            e2 = y == jj
            intersect = np.sum(np.logical_and(e1, e2))
            union = np.sum(np.logical_or(e1, e2))
            cost_matrix[ii, jj] = -float(intersect) / union
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    new_cost_matrix = cost_matrix[row_idx][:, col_idx]
    idxs = np.argsort(np.diag(new_cost_matrix))
    new_cost_matrix = new_cost_matrix[idxs][:, idxs]

    new_y = np.zeros_like(y)
    for ii in range(dim):
        new_y[yk == row_idx[ii]] = col_idx[ii]
    accuracy = (new_y == y).mean()

    null = []
    for ii in range(10):
        null.append(np.equal(y, np.random.permutation(new_y)).mean())
    null_accuracy = np.mean(null)

    return (row_idx, col_idx,
            cost_matrix, new_cost_matrix,
            new_y, accuracy, null_accuracy)
