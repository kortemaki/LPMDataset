import os
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from cyclopts import App


DATA_DIR = os.environ.get('DATASET_DIR', '')

app = App(help="OCR bounding-box visualization tools.")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_ocr_data(fname):
  """Load an *_ocr.csv file and return a cleaned DataFrame.

  The CSV has columns: (unnamed index), level, page_num, block_num, par_num,
  line_num, word_num, left, top, width, height, conf, text.
  Rows with missing/empty text or negative confidence are dropped.
  """
  df = pd.read_csv(fname, index_col=0)
  df['text'] = df['text'].astype(str).str.strip()
  df = df[df['text'].ne('') & df['text'].ne('nan')].reset_index(drop=True)
  return df


def _prepare_plot(image_path, df, *, conf_threshold=0, figsize=(14, 10)):
  """Load image, filter rows, compute OCR-to-image scale factors, create figure."""
  img = plt.imread(image_path)
  img_h, img_w = img.shape[:2]

  fig, ax = plt.subplots(1, figsize=figsize)
  ax.imshow(img)

  visible = df[df['conf'] >= conf_threshold].reset_index(drop=True)

  # Scale from OCR source resolution (1920x1080) to actual image size
  ocr_w, ocr_h = 1920, 1080
  sx = img_w / ocr_w
  sy = img_h / ocr_h

  return fig, ax, visible, sx, sy


_FONT_SIZE = 7
# Approximate pixels-per-character at _FONT_SIZE (proportional font).
_CHAR_WIDTH_PX = _FONT_SIZE * 0.6


def _wrap_label(text, box_width_px):
  """Wrap *text* so no line is wider than *box_width_px*."""
  max_chars = max(1, int(box_width_px / _CHAR_WIDTH_PX))
  return textwrap.fill(text, width=max_chars)


def _draw_boxes(ax, visible, sx, sy, labels=None):
  """Draw bounding boxes coloured by *labels* (or row index) using tab10."""
  cmap = plt.cm.get_cmap('tab10')

  for idx, (_, row) in enumerate(visible.iterrows()):
    label = labels[idx] if labels is not None else idx
    color = cmap(label % cmap.N)
    box_w = row['width'] * sx
    rect = patches.Rectangle(
      (row['left'] * sx, row['top'] * sy),
      box_w, row['height'] * sy,
      linewidth=1.5, edgecolor=color, facecolor='none',
    )
    ax.add_patch(rect)
    wrapped = _wrap_label(str(row['text']), box_w)
    ax.text(
      row['left'] * sx, row['top'] * sy - 2,
      wrapped,
      fontsize=_FONT_SIZE, color=color,
      verticalalignment='bottom',
    )


def _finalize_plot(fig, ax, *, title=None, show=True):
  """Apply common finishing touches and optionally display the figure."""
  ax.set_axis_off()
  if title:
    ax.set_title(title)
  fig.tight_layout()
  if show:
    plt.show()
  return fig, ax


# ---------------------------------------------------------------------------
# Public API (importable)
# ---------------------------------------------------------------------------

def plot_ocr_boxes(image_path, df, *, conf_threshold=0, figsize=(14, 10),
                   title=None, show=True):
  """Overlay OCR bounding boxes onto the source slide image.

  Parameters
  ----------
  image_path : str
      Path to the slide image file (jpg/png).
  df : pd.DataFrame
      OCR dataframe as returned by :func:`load_ocr_data`.
  conf_threshold : int, optional
      Minimum confidence to include a box (default 0 = show all).
  figsize : tuple, optional
      Figure size passed to matplotlib.
  title : str, optional
      Plot title.
  show : bool, optional
      Whether to call ``plt.show()`` (default True).

  Returns
  -------
  fig, ax
      The matplotlib Figure and Axes objects.
  """
  fig, ax, visible, sx, sy = _prepare_plot(
    image_path, df, conf_threshold=conf_threshold, figsize=figsize,
  )
  _draw_boxes(ax, visible, sx, sy)
  return _finalize_plot(fig, ax, title=title, show=show)


def _box_corners(visible):
  """Return an (N, 4, 2) array of bbox corners (TL, TR, BL, BR) in OCR coords."""
  left = visible['left'].values.astype(float)
  top = visible['top'].values.astype(float)
  right = left + visible['width'].values.astype(float)
  bottom = top + visible['height'].values.astype(float)
  return np.stack([
    np.column_stack([left, top]),
    np.column_stack([right, top]),
    np.column_stack([left, bottom]),
    np.column_stack([right, bottom]),
  ], axis=1)


def agglomerative_cluster(corners, n_clusters):
  """Bottom-up (Brown-style) clustering using corner-to-corner distance.

  Each cluster maintains the four outermost corners of its enclosing
  bounding box.  The distance between two clusters is the minimum
  Euclidean distance across all 4x4 corner pairs.  On merge the
  enclosing box is updated to span both clusters.

  Parameters
  ----------
  corners : np.ndarray, shape (N, 4, 2)
      Four (x, y) corners per bounding box.
  n_clusters : int

  Returns
  -------
  labels : np.ndarray of int, shape (N,)
  """
  n = len(corners)
  if n <= n_clusters:
    return np.arange(n)

  # Each box starts as its own cluster.
  cluster_corners = {i: corners[i].copy() for i in range(n)}
  cluster_members = {i: [i] for i in range(n)}
  active = set(range(n))

  def _corner_dist(a, b):
    ca, cb = cluster_corners[a], cluster_corners[b]  # each (4, 2)
    return np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2).min()

  def _merge_corners(a, b):
    all_c = np.concatenate([cluster_corners[a], cluster_corners[b]])  # (8, 2)
    x0, y0 = all_c.min(axis=0)
    x1, y1 = all_c.max(axis=0)
    return np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])

  def _merge_into(target, other):
    cluster_corners[target] = _merge_corners(target, other)
    cluster_members[target].extend(cluster_members[other])
    del cluster_members[other]
    del cluster_corners[other]
    active.remove(other)

  def _absorb_contained(target):
    """Absorb any cluster whose centre lies inside *target*'s bbox."""
    cc = cluster_corners[target]
    x0, y0 = cc.min(axis=0)
    x1, y1 = cc.max(axis=0)
    changed = True
    while changed:
      changed = False
      for other in sorted(active - {target}):
        cx, cy = cluster_corners[other].mean(axis=0)
        if x0 <= cx <= x1 and y0 <= cy <= y1:
          _merge_into(target, other)
          # Bbox may have grown; refresh and restart scan.
          cc = cluster_corners[target]
          x0, y0 = cc.min(axis=0)
          x1, y1 = cc.max(axis=0)
          changed = True
          break

  while len(active) > n_clusters:
    best_dist = float('inf')
    best_pair = None
    active_list = sorted(active)
    for i_pos in range(len(active_list)):
      for j_pos in range(i_pos + 1, len(active_list)):
        ci, cj = active_list[i_pos], active_list[j_pos]
        d = _corner_dist(ci, cj)
        if d < best_dist:
          best_dist = d
          best_pair = (ci, cj)

    ci, cj = best_pair
    _merge_into(ci, cj)
    _absorb_contained(ci)

  # Map surviving cluster IDs to contiguous labels 0 .. k-1.
  labels = np.empty(n, dtype=int)
  for new_label, (_, members) in enumerate(sorted(cluster_members.items())):
    for m in members:
      labels[m] = new_label
  return labels


def _draw_cluster_boxes(ax, visible, sx, sy, labels):
  """Draw one merged bounding box per cluster with concatenated text."""
  cmap = plt.cm.get_cmap('tab10')

  for label in range(labels.max() + 1):
    mask = labels == label
    members = visible[mask]
    if members.empty:
      continue

    x0 = members['left'].min()
    y0 = members['top'].min()
    x1 = (members['left'] + members['width']).max()
    y1 = (members['top'] + members['height']).max()
    text = ' '.join(members['text'].astype(str))
    box_w = (x1 - x0) * sx

    color = cmap(label % cmap.N)
    rect = patches.Rectangle(
      (x0 * sx, y0 * sy),
      box_w, (y1 - y0) * sy,
      linewidth=1.5, edgecolor=color, facecolor='none',
    )
    ax.add_patch(rect)
    wrapped = _wrap_label(text, box_w)
    ax.text(
      x0 * sx, y0 * sy - 2,
      wrapped,
      fontsize=_FONT_SIZE, color=color,
      verticalalignment='bottom',
    )


def plot_ocr_clusters(image_path, df, *, n_clusters=5, conf_threshold=0,
                      figsize=(14, 10), title=None, show=True):
  """Cluster OCR bounding boxes and overlay them colour-coded on the image.

  Parameters
  ----------
  image_path : str
  df : pd.DataFrame
  n_clusters : int
  conf_threshold : int
  figsize : tuple
  title : str | None
  show : bool

  Returns
  -------
  fig, ax, labels
  """
  fig, ax, visible, sx, sy = _prepare_plot(
    image_path, df, conf_threshold=conf_threshold, figsize=figsize,
  )

  corners = _box_corners(visible)
  labels = agglomerative_cluster(corners, n_clusters)

  _draw_cluster_boxes(ax, visible, sx, sy, labels)
  fig, ax = _finalize_plot(fig, ax, title=title, show=show)
  return fig, ax, labels


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@app.command
def boxes(ocr_csv: str, image_path: str, *,
          conf_threshold: int = 0, title: str | None = None):
  """Overlay individual OCR bounding boxes on a slide image."""
  df = load_ocr_data(ocr_csv)
  print(f'{len(df)} OCR entries loaded.')
  plot_ocr_boxes(image_path, df, conf_threshold=conf_threshold,
                 title=title or os.path.basename(ocr_csv))


@app.command
def cluster(ocr_csv: str, image_path: str, *,
            n_clusters: int = 5, conf_threshold: int = 0,
            title: str | None = None):
  """Agglomerative (Brown-style) clustering of OCR bounding boxes.

  Repeatedly merges the two closest clusters (corner-to-corner distance)
  until at most *n_clusters* remain, then visualises the merged boxes.
  """
  df = load_ocr_data(ocr_csv)
  print(f'{len(df)} OCR entries loaded.')
  _, _, labels = plot_ocr_clusters(
    image_path, df, n_clusters=n_clusters,
    conf_threshold=conf_threshold,
    title=title or f'{os.path.basename(ocr_csv)} ({n_clusters} clusters)',
  )
  print(f'Clustered into {len(set(labels))} groups.')


if __name__ == '__main__':
  app()
