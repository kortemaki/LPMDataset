import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.interpolate import make_interp_spline
from scipy.stats import wasserstein_distance_nd
from PIL import Image

from src.lpmdataset.modalities import mouse

DATA_DIR = "mlpdataset\data_oct"
#DATA_DIR = os.environ['DATASET_DIR']

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

    def low_res(self,bins=32) -> np.ndarray:
        hist, xedges, yedges = np.histogram2d(self.traces['x'], self.traces['y'], bins=bins,    # FIXED GLOBAL RANGE
                                               density=True)
        hist += 1e-7  # ensure every bin is non-zero
        return hist, xedges, yedges
    def distance_to(self, other: "HeatMap") -> float:
        u_weights, u_xedges, u_yedges = self.low_res()
        v_weights, v_xedges, v_yedges = other.low_res()
        return wasserstein_distance_nd(
             np.array(np.meshgrid(
                0.5 * (u_xedges[:-1] + u_xedges[1:]),
                0.5 * (u_yedges[:-1] + u_yedges[1:])
            )).reshape(2, -1).T,
            np.array(np.meshgrid(
                0.5 * (v_xedges[:-1] + v_xedges[1:]),
                0.5 * (v_yedges[:-1] + v_yedges[1:])
            )).reshape(2, -1).T,
            u_weights.flatten(),
            v_weights.flatten()
        )



def visualize_attention(slide_path, heatmap, sentence_text, output_path="debug_attention.png"):
    """
    slide_path: Path to the original slide PNG
    heatmap: The (14, 14) tensor/array from get_attention_heatmap
    sentence_text: The ASR sentence being grounded
    """
    # 1. Load the original image
    img = Image.open(slide_path).convert("RGB")
    width, height = img.size

    # 2. Normalize and upsample the heatmap
    # Convert from torch tensor to numpy if necessary
    if hasattr(heatmap, 'detach'):
        heatmap = heatmap.detach().cpu().numpy()

    # Min-max normalization for better contrast in visualization
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # 3. Create the plot
    plt.figure(figsize=(10, 6))
    plt.imshow(img)

    # Overlay the heatmap
    # 'extent' maps the 14x14 grid to the full pixel dimensions of the image
    plt.imshow(heatmap, cmap='jet', alpha=0.4, extent=(0, width, height, 0), interpolation='bilinear')

    plt.title(f"ASR Grounding: {sentence_text[:50]}...")
    plt.axis('off')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    plt.show()



def __main__() -> None:
    for fname in ["anat-1/AnatomyPhysiology/01/slide_001_trace.csv"]:
        df = mouse.load_trace_data(os.path.join(DATA_DIR, fname))
        hm = HeatMap(df)
        hm.upsample()
        hm.show(title=fname, bins=32)

        print(hm.distance_to(hm))
        print(hm.distance_to(mouse.load_trace_data(os.path.join(DATA_DIR, "anat-1/AnatomyPhysiology/01/slide_002_trace.csv"))))
        print(hm.distance_to(mouse.load_trace_data(os.path.join(DATA_DIR, "anat-1/AnatomyPhysiology/01/slide_006_trace.csv"))))


if __name__=="__main__":
    __main__()