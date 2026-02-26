import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.interpolate import make_interp_spline
from scipy.stats import wasserstein_distance_nd

from lpmdataset.modalities import mouse


DATA_DIR = os.environ['DATASET_DIR']

class HeatMap:
    def __init__(self, df):
        self.traces = df

    def upsample(self, pause_threshold=0.5) -> None:
        df = self.traces.sort_values('timestamp').reset_index(drop=True)

        # Identify motifs: consecutive points with <= pause_threshold gap
        gaps = df['timestamp'].diff()
        motif_ids = (gaps > pause_threshold).cumsum()

        upsampled_parts = []

        for _, group in df.groupby(motif_ids):
            group = group.reset_index(drop=True)
            coords = group[['x', 'y']].values
            timestamps = group['timestamp'].values

            if len(group) < 2:
                upsampled_parts.append(group[['timestamp', 'x', 'y']])
                continue

            # Compute cord length (cumulative sum of Euclidean distances)
            diffs = np.diff(coords, axis=0)
            seg_lengths = np.linalg.norm(diffs, axis=1)
            s = np.concatenate([[0.0], np.cumsum(seg_lengths)])

            # Skip motifs where the mouse didn't move
            if s[-1] == 0:
                upsampled_parts.append(group[['timestamp', 'x', 'y']])
                continue

            # Remove duplicate cord lengths (stationary consecutive points)
            unique_mask = np.concatenate([[True], np.diff(s) > 0])
            s_unique = s[unique_mask]
            coords_unique = coords[unique_mask]

            if len(s_unique) < 2:
                upsampled_parts.append(group[['timestamp', 'x', 'y']])
                continue

            # Fit splines parameterized by cord length
            k = min(3, len(s_unique) - 1)
            spline_x = make_interp_spline(s_unique, coords_unique[:, 0], k=k)
            spline_y = make_interp_spline(s_unique, coords_unique[:, 1], k=k)

            # Generate new timestamps at 1ms intervals
            t_start, t_end = timestamps[0], timestamps[-1]
            n_points = int(round((t_end - t_start) / 0.001)) + 1
            new_timestamps = np.linspace(t_start, t_end, n_points)

            # Map new timestamps to cord lengths via linear interpolation
            new_s = np.interp(new_timestamps, timestamps, s)

            upsampled_parts.append(pd.DataFrame({
                'timestamp': new_timestamps,
                'x': spline_x(new_s),
                'y': spline_y(new_s),
            }))

        self.traces = pd.concat(upsampled_parts, ignore_index=True)

    def show(self, *, bins=224, title=None, norm=colors.LogNorm()) -> None:
        hist, xedges, yedges = np.histogram2d(self.traces['x'], self.traces['y'], bins=bins, density=True)
        hist += 1e-7  # ensure every bin is non-zero

        plt.imshow(
            hist.T, origin='lower', aspect='auto',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            norm=norm
        )
        plt.colorbar(label='Counts (log scale)')
        if title:
            plt.title(title)
        plt.show()

    def low_res(self) -> np.ndarray:
        hist, xedges, yedges = np.histogram2d(self.traces['x'], self.traces['y'], bins=32, density=True)
        hist += 1e-7  # ensure every bin is non-zero
        return hist, xedges, yedges

    def distance_to(self, other: "HeatMap") -> float:
        u_weights, u_xedges, u_yedges = self.low_res()
        v_weights, v_xedges, v_yedges = other.low_res()
        return wasserstein_distance_nd(
            list(zip(*(dim.tolist() for dim in np.meshgrid((u_xedges, u_yedges))))),
            list(zip(*(dim.tolist() for dim in np.meshgrid((v_xedges, v_yedges))))),
            u_weights, v_weights
        )


def __main__() -> None:
    for fname in ["anat-1/AnatomyPhysiology/13/slide_004_trace.csv"]:
        df = mouse.load_trace_data(os.path.join(DATA_DIR, fname))
        hm = HeatMap(df)
        hm.upsample()
        hm.show(title=fname, bins=224)

        print(hm.distance_to(hm))
        print(hm.distance_to(mouse.load_trace_data(os.path.join(DATA_DIR, "anat-1/AnatomyPhysiology/01/slide_002_trace.csv"))))
        print(hm.distance_to(mouse.load_trace_data(os.path.join(DATA_DIR, "anat-1/AnatomyPhysiology/01/slide_006_trace.csv"))))


if __name__=="__main__":
    __main__()
