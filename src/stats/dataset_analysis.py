import ast
from collections import Counter, defaultdict
import csv
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

DATA_DIR = os.environ['DATASET_DIR']
FIGURES_DIR = os.environ['FIGURES_DIR']
FOLDERS = {
    'dental/OperativeDentistryNBDEPartII',
    'dental/PediatricDentistryNBDEPartII',
    'dental/Evidence-BasedDentistry',
    'dental/EndodonticsNBDEPartII',
    'psy-2/LectureSeriesforIntrotoDevelopmentalPsy',
    'dental/OralMedicineINBDE',
    'dental/Orthodontics',
    'dental/ProsthodonticsNBDEPartII',
    'dental/NBDEPartI',
    'anat-1/AnatomyPhysiology',
    'psy-1/LectureSeriesForIntrotoPsy-PSY101',
    'dental/OralSurgeryNBDEPartII',
    'dental/OralPathologyNBDEPartII',
    'bio-1/unordered',
    'speaking/EssayWritingandPresentationskills',
    'dental/PeriodonticsNBDEPartII',
    'bio-4/unordered',
    'dental/Cariology',
    'dental/OralRadiologyNBDEPartII',
    'dental/TheBasicsofDentistry',
    'dental/PatientManagementNBDEPartII',
    'bio-3/Biol1020',
    'dental/PerceptualAbilityTestDAT',
    'ml-1/MultimodalMachineLearning',
    'dental/OrthodonticsNBDEPartII',
    'anat-2/unordered',
    'dental/HeadNeckAnatomyINBDE',
    'dental/Occlusion'
}
SPEAKER_RESOLUTIONS = {
  'dental/OperativeDentistryNBDEPartII': '720p',
  'dental/PediatricDentistryNBDEPartII': '1080p',
  'dental/Evidence-BasedDentistry': '720p',
  'dental/EndodonticsNBDEPartII': '1080p',
  'psy-2/LectureSeriesforIntrotoDevelopmentalPsy': '720p',
  'dental/OralMedicineINBDE': '1080p',
  'dental/Orthodontics': '720p',
  'dental/ProsthodonticsNBDEPartII': '1080p',
  'dental/NBDEPartI': '720p',
  'anat-1/AnatomyPhysiology': '1080p',
  'psy-1/LectureSeriesForIntrotoPsy-PSY101': '720p',
  'dental/OralSurgeryNBDEPartII': '1080p',
  'dental/OralPathologyNBDEPartII': '1080p',
  'bio-1/unordered': '1080p',
  'speaking/EssayWritingandPresentationskills': '480p',
  'dental/PeriodonticsNBDEPartII': '1080p',
  'bio-4/unordered': '720p',
  'dental/Cariology': '480p',
  'dental/OralRadiologyNBDEPartII': '720p',
  'dental/TheBasicsofDentistry': '1080p',
  'dental/PatientManagementNBDEPartII': '1080p',
  'bio-3/Biol1020': '480p',
  'dental/PerceptualAbilityTestDAT': '720p',
  'ml-1/MultimodalMachineLearning': 'SXGA',
  'dental/OrthodonticsNBDEPartII': '720p',
  'anat-2/unordered': '480p',
  'dental/HeadNeckAnatomyINBDE': '720p',
  'dental/Occlusion': '480p',
}

RESOLUTIONS = {
  '240p': (426, 240),
  '360p': (640, 360),
  '480p': (854, 480),
  '720p': (1280, 720),
  '1080p': (1920, 1080),
  '1440p': (2560, 1440),
  'SXGA': (1280, 1024),
}

def get_asr_counts():
  asr_counts = Counter()
  for folder in FOLDERS:
    for subdir in [_ for _ in os.listdir(os.path.join(DATA_DIR, folder)) if os.path.isdir(os.path.join(DATA_DIR, folder, _))]:
      for f in [_ for _ in os.listdir(os.path.join(DATA_DIR, folder, subdir)) if _[-11:]=='_spoken.csv']:
        asr_counts.update([tok for l in csv.reader(open(os.path.join(DATA_DIR, folder, subdir, f), 'r', encoding='utf-8')) for tok in l[1].split(' ')])
  return asr_counts

def get_ocr_counts():
    ocr_counts = Counter()
    for folder in FOLDERS:
      for subdir in [_ for _ in os.listdir(os.path.join(DATA_DIR, folder)) if os.path.isdir(os.path.join(DATA_DIR, folder, _))]:
        for f in [_ for _ in os.listdir(os.path.join(DATA_DIR, folder, subdir)) if _[-8:]=='_ocr.csv']:
          ocr_counts.update([tok for l in csv.reader(open(os.path.join(DATA_DIR, folder, subdir, f), 'r', encoding='utf-8')) for tok in l[-1].split(' ')])
    return ocr_counts

def get_file_counts_and_sizes():
  file_counts = Counter()
  total_sizes = Counter()
  for folder in FOLDERS:
    for subdir in [_ for _ in os.listdir(os.path.join(DATA_DIR, folder)) if os.path.isdir(os.path.join(DATA_DIR, folder, _))]:
      file_counts.update([_.split('_')[-1] for _ in os.listdir(os.path.join(DATA_DIR, folder, subdir)) if _[:6]=='slide_' and _[-4:]=='.csv'])
      for f in [_ for _ in os.listdir(os.path.join(DATA_DIR, folder, subdir)) if _[:6]=='slide_' and _[-4:]=='.csv']:
        total_sizes[f.split('_')[-1]] += os.path.getsize(os.path.join(DATA_DIR, folder, subdir, f))

  return file_counts, total_sizes


lens = [float(_) for _ in [l[2] for l in csv.reader(open(os.path.join(DATA_DIR, "raw_video_links.csv"), 'r', encoding='utf-8'))][1:]]
slide_times = [l[1].split('|') for l in csv.reader(open(os.path.join(DATA_DIR, "raw_video_links.csv"), 'r', encoding='utf-8'))][1:]


def make_word_clouds():
  # word clouds
  for folder in FOLDERS:
    for subdir in [_ for _ in os.listdir(os.path.join(DATA_DIR, folder)) if os.path.isdir(os.path.join(DATA_DIR, folder, _))]:
      for f in [_ for _ in os.listdir(os.path.join(DATA_DIR, folder, subdir)) if _[-11:]=='_spoken.csv']:
        asr_text_all.append([tok for l in csv.reader(open(os.path.join(DATA_DIR, folder, subdir, f), 'r', encoding='utf-8')) for tok in l[1].split(' ')])
      for f in [_ for _ in os.listdir(os.path.join(DATA_DIR, folder, subdir)) if _[-8:]=='_ocr.csv']:
        ocr_text_all.append([tok for l in csv.reader(open(os.path.join(DATA_DIR, folder, subdir, f), 'r', encoding='utf-8')) for tok in l[-1].split(' ')])
  asr_text_all = [' '.join([w for w in text[1:] if w not in STOPWORDS]) for text in asr_text_all]

  wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(asr_text_all))
  plt.figure(figsize=(10, 5))
  plt.imshow(wc, interpolation='bilinear')
  plt.axis('off')
  plt.savefig(os.path.join(FIGURES_DIR, 'ASR_wordcloud.png'))


def make_bounding_box_stats():
  # fig bounding boxes
  df = pd.read_csv(os.path.join(DATA_DIR, "figure_annotations.csv"))
  df['boundingBoxes'] = df['boundingBoxes'].apply(json.loads)
  figs = df.explode('boundingBoxes').dropna()
  figs = pd.concat(
    [
      figs[['Input.save_dir']].reset_index(drop=True),
      pd.json_normalize(figs['boundingBoxes']).reset_index(drop=True),
    ],
    axis=1,
  )
  figs['width'] = figs.apply(lambda row: row['width']/RESOLUTIONS[SPEAKER_RESOLUTIONS['/'.join(row['Input.save_dir'][5:].split('/')[:2])]][0], axis=1)
  figs['height'] = figs.apply(lambda row: row['height']/RESOLUTIONS[SPEAKER_RESOLUTIONS['/'.join(row['Input.save_dir'][5:].split('/')[:2])]][1], axis=1)
  sns.set_theme(style='white')
  g = sns.JointGrid(data=figs, x='width', y='height', hue='label')
  g.plot_joint(sns.scatterplot, alpha=0.5, s=15, hue_order=['Diagram', 'Image', 'Equation', 'Table'])
  g.plot_marginals(sns.kdeplot, fill=True, alpha=0.3, common_norm=False)
  g.ax_joint.set_xlim(0, 1.05)
  g.ax_joint.set_ylim(0, 1.05)
  handles, labels = g.ax_joint.get_legend_handles_labels()
  g.ax_joint.legend(handles[:4], labels[:4], title="Figure Type")
  plt.tight_layout()
  plt.savefig(os.path.join(FIGURES_DIR, 'BoundingBoxDist.png'), dpi=600)


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

def get_max_mouse_points():
  min_x = defaultdict(lambda: 1000)
  min_y = defaultdict(lambda: 1000)
  max_x = defaultdict(lambda: 0)
  max_y = defaultdict(lambda: 0)
  for folder in FOLDERS:
    for subdir in [_ for _ in os.listdir(os.path.join(DATA_DIR, folder)) if os.path.isdir(os.path.join(DATA_DIR, folder, _))]:
      lecture = os.path.joni(DATA_DIR, folder, subdir)
      for f in [_ for _ in os.listdir(os.path.join(DATA_DIR, folder, subdir)) if _[:6]=='slide_' and _[-10:]=='_trace.csv']:
        t = load_trace_data(os.path.join(DATA_DIR, folder, subdir, f))
        if t.shape[0]:
          min_x[lecture] = min(t.x.min(), min_x[lecture])
          min_y[lecture] = min(t.y.min(), min_y[lecture])
          max_x[lecture] = min(t.x.max(), max_x[lecture])
          max_y[lecture] = min(t.y.max(), max_y[lecture])
  return {
    folder: (
      max(v_x for fname, v_x in max_x.items() if fname.startswith(os.path.join(DATA_DIR, folder))) + 1,
      max(v_y for fname, v_y in max_y.items() if fname.startswith(os.path.join(DATA_DIR, folder))) + 1)
    for folder in folders
  }

def compute_gestures_and_path_lengths():
  fnames = []
  gestures = []
  path_lengths = []
  path_lengths_sums = []
  for folder in FOLDERS:
    for subdir in [_ for _ in os.listdir(os.path.join(DATA_DIR, folder)) if os.path.isdir(os.path.join(DATA_DIR, folder, _))]:
      for f in [_ for _ in os.listdir(os.path.join(DATA_DIR, folder, subdir)) if _[:6]=='slide_' and _[-10:]=='_trace.csv']:
        stats = analyze_slide_mouse_data(load_trace_data(os.path.join(DATA_DIR, folder, subdir, f)))
        fnames.append(os.path.join(DATA_DIR, folder, subdir, f))
        gestures.append(stats.shape[0])
        path_lengths.append(stats.path_length.tolist() if stats.shape[0] else [])
        path_lengths_sums.append(stats.path_length.sum() if stats.shape[0] else 0.0)
  return pd.DataFrame(list(zip(fnames, gestures, path_lengths, path_lengths_sums)), columns=['fname', 'n_gestures', 'path_lengths', 'total_path_lengths'])

def get_mouse_gesture_flows():
  gesture_moments = []
  for folder in FOLDERS:
    X, Y = RESOLUTIONS[SPEAKER_RESOLUTIONS[folder]]
    for subdir in [_ for _ in os.listdir(os.path.join(DATA_DIR, folder)) if os.path.isdir(os.path.join(DATA_DIR, folder, _))]:
      for f in [_ for _ in os.listdir(os.path.join(DATA_DIR, folder, subdir)) if _[:6]=='slide_' and _[-10:]=='_trace.csv']:
        for _, row in analyze_slide_mouse_data(load_trace_data(os.path.join(DATA_DIR, folder, subdir, f))).iterrows():
          gesture_moments.append([row.p_start[0]/X, row.p_start[1]/Y, (row.p_end[0] - row.p_start[0])/X/row.duration, (row.p_end[1] - row.p_start[1])/Y/row.duration])

  return pd.DataFrame(gesture_moments, columns=['x', 'y', 'dx', 'dy'])

def plot_mouse_gesture_flows(df, grid_res=50):
  xi = np.linspace(0, 1, grid_res)
  yi = np.linspace(0, 1, grid_res)
  X, Y = np.meshgrid(xi, yi)
  U = griddata((df['x'], df['y']), df['dx'], (X, Y), method='linear', fill_value=0)
  V = griddata((df['x'], df['y']), df['dy'], (X, Y), method='linear', fill_value=0)

  speed = np.sqrt(U**2 + V**2)
  v_min = 0.1
  v_max = speed.max()

  lw = 1.0 + 5.0 * speed / v_max

  plt.figure(figsize=(12, 7), facecolor='black')
  norm = mcolors.LogNorm(vmin=v_min, vmax=v_max)
  strm = plt.streamplot(
    X, Y, U, V,
    color=speed,
    linewidth=lw,
    cmap='plasma',
    density=3.0,
    arrowstyle='->',
    arrowsize=1.5,
    norm=norm,
  )

  cbar = plt.colorbar(strm.lines, norm=norm)
  cbar.set_label('Mouse Speed (screen lengths per second)', color='white')
  cbar.ax.yaxis.set_tick_params(labelcolor='white')

  plt.gca().invert_yaxis()
  plt.axis('off')
  plt.tight_layout()
  plt.savefig(os.path.join(FIGURES_DIR, "MouseGestureFlows.png"), dpi=600)
  plt.title("Mouse Movement Flow Field", color='white')
  plt.show()
