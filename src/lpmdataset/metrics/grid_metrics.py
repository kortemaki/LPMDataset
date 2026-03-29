import numpy as np
from scipy.stats import wasserstein_distance_nd


def heatmap_to_distribution(heatmap):
    grid_size = heatmap.shape[0]
    xs = np.linspace(0, 1, grid_size)
    ys = np.linspace(0, 1, grid_size)

    coords = []
    weights = []

    for i in range(grid_size):
        for j in range(grid_size):
            coords.append([xs[j], ys[i]])
            weights.append(heatmap[i, j])

    coords = np.array(coords)
    weights = np.array(weights)

    weights = weights / (weights.sum() + 1e-8)

    return coords, weights


def heatmap_to_mle(heatmap):
    """Return the (x, y) coordinate of the bucket with the maximum activation.

    Parameters
    ----------
    heatmap : np.ndarray
        Unnormalized 2-D activation grid

    Returns
    -------
    np.ndarray
        Shape-(2,) array ``[x, y]`` in [0, 1] corresponding to the center of
        the highest-activation bucket.

    Notes
    -----
    Tie-breaking: when multiple buckets share the maximum value, the one whose
    **flattened (row-major) index is smallest** is chosen.  This is the
    top-left-most winner when scanning left-to-right, top-to-bottom — i.e.
    ``np.argmax`` behaviour on the flattened array.
    """
    rows, cols = heatmap.shape
    xs = np.linspace(0, 1, cols)
    ys = np.linspace(0, 1, rows)

    # np.argmax returns the first (row-major) occurrence on ties
    flat_idx = np.argmax(heatmap)
    i, j = np.unravel_index(flat_idx, heatmap.shape)

    return np.array([xs[j], ys[i]])


def compute_iou(pred_df, gt_df, grid_size):
    """
    Updated to accept dynamic grid_size (e.g., 14 for LayoutLMv3, 16 for CLIP)
    to ensure spatial bins perfectly align with the model's native resolution.
    """
    hist1, _, _ = np.histogram2d(
        pred_df["x"], pred_df["y"],
        bins=grid_size,
        range=[[0,1],[0,1]]
    )

    hist2, _, _ = np.histogram2d(
        gt_df["x_normalized"], gt_df["y_normalized"],
        bins=grid_size,
        range=[[0,1],[0,1]]
    )

    hist1 /= hist1.sum() + 1e-8
    hist2 /= hist2.sum() + 1e-8

    intersection = np.minimum(hist1, hist2).sum()
    union = np.maximum(hist1, hist2).sum()

    return intersection / (union + 1e-8)


def compute_rmse(heatmap, mouse_df):
    grid_size = heatmap.shape[0]
    xs = np.linspace(0, 1, grid_size)
    ys = np.linspace(0, 1, grid_size)

    exp_x = 0
    exp_y = 0

    for i in range(grid_size):
        for j in range(grid_size):
            exp_x += xs[j] * heatmap[i, j]
            exp_y += ys[i] * heatmap[i, j]

    gt_x = mouse_df["x_normalized"].mean()
    gt_y = mouse_df["y_normalized"].mean()

    return np.sqrt((exp_x - gt_x)**2 + (exp_y - gt_y)**2)


def compute_wasserstein(heatmap, mouse_df):
    pred_coords, pred_weights = heatmap_to_distribution(heatmap)

    gt_coords = mouse_df[["x_normalized","y_normalized"]].values
    gt_weights = np.ones(len(gt_coords))
    gt_weights /= gt_weights.sum()

    return wasserstein_distance_nd(
        pred_coords,
        gt_coords,
        pred_weights,
        gt_weights
    )
