import ast

import numpy as np
import pandas as pd


def calculate_mad(points: np.ndarray) -> float:
  """
  Calculates the Maximum Absolute Deviation (MAD) for a 2D trajectory.
  Points is a numpy array of shape (N, 2) -> [[x1, y1], [x2, y2], ...]
  """
  if len(points) < 3:
    return 0.0

  p_start = points[0]
  p_end = points[-1]

  # Check if start and end are the same point to avoid division by zero
  if np.array_equal(p_start, p_end):
    # If they are the same, MAD is the max distance from this point
    return np.max(np.linalg.norm(points - p_start, axis=1))

  # Vectorized perpendicular distance formula
  # Line defined by p_start and p_end: ax + by + c = 0
  a = p_start[1] - p_end[1]
  b = p_end[0] - p_start[0]
  c = (p_start[0] * p_end[1]) - (p_end[0] * p_start[1])

  distances = np.abs(a * points[:, 0] + b * points[:, 1] + c) / np.sqrt(a**2 + b**2)
  return np.max(distances)

def analyze_slide_mouse_data(df, pause_threshold=0.5):
  """
  Processes a dataframe of [timestamp, x, y] for a single slide.
  - Identifies segments separated by pauses.
  - Calculates MAD and total travel distance per segment.
  """
  # 1. Calculate time gaps to identify distinct movements
  df = df.sort_values('timestamp')
  df['gap'] = df['timestamp'].diff()

  # 2. Assign Segment IDs (new segment starts if gap > threshold)
  df['segment_id'] = (df['gap'] > pause_threshold).cumsum()

  results = []

  for seg_id, group in df.groupby('segment_id'):
    if len(group) < 2:
      continue

    coords = group[['x', 'y']].values

    # Calculate Segment Stats
    mad_val = calculate_mad(coords)

    # Path Length (cumulative distance between points)
    diffs = np.diff(coords, axis=0)
    path_length = np.sum(np.sqrt((diffs**2).sum(axis=1)))

    # Displacement (straight line from start to end)
    displacement = np.linalg.norm(coords[-1] - coords[0])

    results.append({
      'segment_id': seg_id,
      'duration': group['timestamp'].max() - group['timestamp'].min(),
      'p_start': coords[0],
      'p_end': coords[-1],
      'path_length': path_length,
      'mad': mad_val,
      'straightness': displacement / path_length if path_length > 0 else 1,
    })

  return pd.DataFrame(results)

def load_trace_data(fname):
  df = pd.read_csv(fname)
  df[['x', 'y']] = pd.DataFrame(df['coord'].apply(ast.literal_eval).tolist(), index=df.index) if df.shape[0] else 0
  return df.rename(columns={'time': 'timestamp'})[['timestamp', 'x', 'y']]
