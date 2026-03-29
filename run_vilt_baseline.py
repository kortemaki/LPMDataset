import os
from time import sleep

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from lpmdataset import data_models
from lpmdataset.metrics import grid_metrics
from lpmdataset.models.baselines.vilt_b32_finetuned_vqa import predict


def process_presentation(p: data_models.Presentation) -> None:
    resolution = data_models.SPEAKER_RESOLUTIONS[p.folder]
    width = resolution.width
    height = resolution.height

    # Build output directory mirroring ocr_asr convention:
    # results/layoutlmv3/<root>/<order>/<video>/<slide>.csv
    folder_parts = str(p.folder).split('/')
    root_folder = folder_parts[0]
    order_folder = folder_parts[1] if len(folder_parts) > 1 else ''
    video_folder = os.path.basename(p.dir_path)
    out_dir = os.path.join("results", "vilt_b32_finetuned_vqa", root_folder, order_folder, video_folder)
    os.makedirs(out_dir, exist_ok=True)

    for s in tqdm(p.slides(), total=sum(1 for _ in p.slides())):
        slide_name = f"slide_{s.slide_no:03d}"
        out_path = os.path.join(out_dir, f"{slide_name}.csv")
        if os.path.exists(out_path):
            continue

        while not os.path.exists(s.asr_text.sentences_path):
            print("Waiting for sentence segmentation")
            sleep(5)

        pred_xs, pred_ys, gold_xs, gold_ys = [], [], [], []

        for tbounds, sent, heatmap in predict(s):
            mouse_seg = s.mouse_trace.between(*tbounds)
            if len(mouse_seg) == 0:
                continue

            xy_pred = grid_metrics.heatmap_to_mle(heatmap)

            # Convert normalised [0,1] prediction to pixel coordinates
            pred_x_px = xy_pred[0] * width
            pred_y_px = xy_pred[1] * height

            # Repeat the single sentence-level prediction for every
            # gold mouse point that falls inside this sentence window
            n_pts = len(mouse_seg)
            pred_xs.extend([pred_x_px] * n_pts)
            pred_ys.extend([pred_y_px] * n_pts)
            gold_xs.extend(mouse_seg['x'].values.tolist())
            gold_ys.extend(mouse_seg['y'].values.tolist())

        if len(pred_xs) == 0:
            continue

        N = len(pred_xs)

        save_df = pd.DataFrame({
            "time": np.arange(N),
            "pred_x": pred_xs,
            "pred_y": pred_ys,
            "gold_x": gold_xs,
            "gold_y": gold_ys,
        })
        save_df.to_csv(out_path, index=False)

if __name__ == '__main__':
    folders_to_process = [data_models.Folder.ML_1, data_models.Folder.SPEAKING]
    for folder in tqdm(
        folders_to_process,
        total=len(folders_to_process)
    ):
        for p in tqdm(
            data_models.iter_presentations(folder),
            total=sum(1 for _ in data_models.iter_presentations(folder))
        ):
            process_presentation(p)
