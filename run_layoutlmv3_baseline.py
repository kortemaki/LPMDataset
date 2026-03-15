import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from lpmdataset.data_models import Presentation, Folder
from lpmdataset.metrics import grid_metrics
from lpmdataset.models.baselines.layoutlmv3_base_finetuned_rvlcdip import predict


if __name__ == '__main__':
    p = Presentation(folder=Folder.ANAT_2, yt_id="ALRJCeVT0fQ")
    presentation_ious = []
    presentation_emds = []
    presentation_rmses = []
    for s in tqdm(p.slides(), total=sum(1 for _ in p.slides())):
        i = Image.open(s.png).convert("RGB")
        s.mouse_trace.normalize_traces(height=900, width=1200)

        slide_ious = []
        slide_emds = []
        slide_rmses = []
        for tbounds, sent, heatmap in predict(s):
            mouse_seg = s.mouse_trace.between(*tbounds)
            if len(mouse_seg) == 0:
                continue

            grid_size = heatmap.shape[0]

            pred_coords, _ = grid_metrics.heatmap_to_distribution(heatmap)

            pred_df = pd.DataFrame({
                "x": pred_coords[:, 0],
                "y": pred_coords[:, 1],
            })

            slide_ious.append(grid_metrics.compute_iou(pred_df, mouse_seg, grid_size=grid_size))
            slide_emds.append(grid_metrics.compute_wasserstein(heatmap, mouse_seg))
            slide_rmses.append(grid_metrics.compute_rmse(heatmap, mouse_seg))
        presentation_ious.append(np.mean(slide_ious))
        presentation_emds.append(np.mean(slide_emds))
        presentation_rmses.append(np.mean(slide_rmses))
    print(f"Slides processed: {len(presentation_ious)}")
    print(f"Average IoU: {np.mean(presentation_ious).mean()}")
    print(f"Average EMD: {np.mean(presentation_emds).mean()}")
    print(f"Average RMSE: {np.mean(presentation_rmses).mean()}")
