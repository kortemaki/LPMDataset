from collections.abc import Iterable
import os
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from cyclopts import App
import pandas as pd
from pydantic import computed_field, ConfigDict
from pydantic.dataclasses import dataclass

from lpmdataset.utils import STOPWORDS


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


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class OCR:
    df: pd.DataFrame

    @computed_field
    def tokens(self) -> Iterable[str]:
        return self.df['text'].tolist()

    def to_string(self) -> str:
        return " ".join(self.tokens)

    @computed_field
    def bbs(self) -> pd.DataFrame:
        return self.df[['left', 'top', 'width', 'height']]

    def rescale_bbs_xyxy(self, height: int, width: int, scale_factor=1000, astype=int) -> None:
        """
        Transforms OCR [left, top, width, height] into [x0, y0, x1, y1]
        scaled to the [0, scale_factor] range using signed integers.
        """
        # 1. Calculate the 4 corners in pixel space
        x0 = np.clip(self.df['left'].values, 0, width)
        y0 = np.clip(self.df['top'].values, 0, height)
        x1 = np.clip(x0 + self.df['width'].values, 0, width)
        y1 = np.clip(y0 + self.df['height'].values, 0, height)

        # 2. Scale and Clamp
        # We use scale_factor - 1 (999) to be absolutely safe against OOB indices
        limit = scale_factor - 1

        s_x0 = np.clip((x0 / width * limit), 0, limit).astype(astype)
        s_y0 = np.clip((y0 / height * limit), 0, limit).astype(astype)
        s_x1 = np.clip((x1 / width * limit), 0, limit).astype(astype)
        s_y1 = np.clip((y1 / height * limit), 0, limit).astype(astype)

        # 3. Stack into the final (N, 4) shape
        self.df['scaled_box'] = np.stack((s_x0, s_y0, s_x1, s_y1), axis=1).tolist()

    def prune_ocr_to_budget(self, tokenizer, token_budget=100):
        """
        1. Removes stopwords.
        2. Calculates token length per word.
        3. Prunes by bounding box area until within token_budget.
        """
        # Define basic stopwords (or use a library like NLTK)
        STOPWORDS = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "of"}

        # 1. Basic Cleaning
        self.df = self.df[~self.df['text'].str.lower().isin(STOPWORDS)].copy()

        # 2. Calculate Token Lengths
        # Note: LayoutLMv3 tokenizer usually adds a prefix space for consistency
        def get_token_len(text):
            # We encode without special tokens to get the raw word count
            tokens = tokenizer._tokenizer.encode(" " + text, add_special_tokens=False)
            return len(tokens.ids)

        self.df['token_count'] = self.df['text'].apply(get_token_len)

        # 3. Calculate "Importance" (Area)
        # Larger text is usually more semantically important on a slide
        self.df['area'] = self.df['width'] * self.df['height']

        # 4. Pruning Loop
        # Sort by area descending (keep the biggest boxes)
        self.df = self.df.sort_values(by='area', ascending=False)

        kept_indices = []
        current_token_total = 0

        for idx, row in self.df.iterrows():
            if current_token_total + row['token_count'] <= token_budget:
                kept_indices.append(idx)
                current_token_total += row['token_count']
            else:
                continue # Skip this word, it's too 'expensive' for the remaining budget

        # Return the pruned dataframe, re-sorted by original document order (top-to-bottom)
        self.df = self.df.loc[kept_indices].copy().sort_index()


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
