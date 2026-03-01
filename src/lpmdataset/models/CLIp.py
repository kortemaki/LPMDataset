import torch
import open_clip
import numpy as np
import pandas as pd
import os
from PIL import Image
from scipy.stats import wasserstein_distance_nd

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# LOAD CLIP
# =========================================================

model, preprocess, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

model = model.to(device)
model.eval()

# =========================================================
# ASR SEGMENTS (2 second window)
# =========================================================

def build_segments(asr_path, window=2.0):

    df = pd.read_csv(asr_path)
    df["Start"] = df["Start"].clip(lower=0)

    segments = []
    t = df["Start"].min()
    end_time = df["End"].max()

    while t < end_time:

        chunk = df[(df["Start"] >= t) & (df["End"] <= t + window)]

        if len(chunk) > 0:
            text = " ".join(chunk["Word"].astype(str).tolist())
            segments.append({
                "start": t,
                "end": t + window,
                "text": text
            })

        t += window

    return segments

# =========================================================
# LOAD MOUSE
# =========================================================

def load_mouse(mouse_path):

    df = pd.read_csv(mouse_path)

    df[["x","y"]] = df["coord"].str.extract(r"\((\d+),\s*(\d+)\)")
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)

    return df

def normalize_mouse(df, screen_w=1200, screen_h=900):
    df["x"] /= screen_w
    df["y"] /= screen_h
    return df

# =========================================================
# PATCH EXTRACTION
# =========================================================

def extract_patches(image, grid_size=16):

    W, H = image.size
    patch_w = W // grid_size
    patch_h = H // grid_size

    patches = []

    for r in range(grid_size):
        for c in range(grid_size):
            left = c * patch_w
            top = r * patch_h
            right = left + patch_w
            bottom = top + patch_h

            patch = image.crop((left, top, right, bottom))
            patches.append(patch)

    return patches

# =========================================================
# ENCODE PATCHES
# =========================================================

def encode_patches(patches):

    processed = []

    for p in patches:
        out = preprocess(p)
        if isinstance(out, list):
            out = out[0]
        processed.append(out)

    image_inputs = torch.stack(processed).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_inputs)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features

# =========================================================
# TEXT → HEATMAP
# =========================================================

def text_to_heatmap(text, image_features, grid_size=16):

    text_tokens = tokenizer([text]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    similarity = image_features @ text_features.T
    similarity = similarity.squeeze()

    heatmap = torch.softmax(similarity, dim=0)
    heatmap = heatmap.reshape(grid_size, grid_size)

    return heatmap.cpu().numpy()

# =========================================================
# HEATMAP TO COORD + WEIGHTS
# =========================================================

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

# =========================================================
# IOU
# =========================================================

def compute_iou(pred_df, gt_df, bins=16):

    hist1, _, _ = np.histogram2d(
        pred_df["x"], pred_df["y"],
        bins=bins,
        range=[[0,1],[0,1]]
    )

    hist2, _, _ = np.histogram2d(
        gt_df["x"], gt_df["y"],
        bins=bins,
        range=[[0,1],[0,1]]
    )

    hist1 /= hist1.sum() + 1e-8
    hist2 /= hist2.sum() + 1e-8

    intersection = np.minimum(hist1, hist2).sum()
    union = np.maximum(hist1, hist2).sum()

    return intersection / (union + 1e-8)

# =========================================================
# RMSE (centroid difference)
# =========================================================

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

    gt_x = mouse_df["x"].mean()
    gt_y = mouse_df["y"].mean()

    return np.sqrt((exp_x - gt_x)**2 + (exp_y - gt_y)**2)

# =========================================================
# WASSERSTEIN
# =========================================================

def compute_wasserstein(heatmap, mouse_df):

    pred_coords, pred_weights = heatmap_to_distribution(heatmap)

    gt_coords = mouse_df[["x","y"]].values
    gt_weights = np.ones(len(gt_coords))
    gt_weights /= gt_weights.sum()

    return wasserstein_distance_nd(
        pred_coords,
        gt_coords,
        pred_weights,
        gt_weights
    )

# =========================================================
# SLIDE EVALUATION
# =========================================================

def evaluate_clip_slide(slide_path, asr_path, mouse_path):

    image = Image.open(slide_path).convert("RGB")

    patches = extract_patches(image)
    image_features = encode_patches(patches)

    segments = build_segments(asr_path)
    mouse_df = load_mouse(mouse_path)

    all_ious = []
    all_emds = []
    all_rmses = []

    for seg in segments:

        mouse_seg = mouse_df[
            (mouse_df["time"] >= seg["start"]) &
            (mouse_df["time"] <= seg["end"])
        ].copy()

        if len(mouse_seg) == 0:
            continue

        mouse_seg = normalize_mouse(mouse_seg)

        heatmap = text_to_heatmap(seg["text"], image_features)

        pred_coords, _ = heatmap_to_distribution(heatmap)

        pred_df = pd.DataFrame({
            "x": pred_coords[:,0],
            "y": pred_coords[:,1]
        })

        iou = compute_iou(pred_df, mouse_seg)
        emd = compute_wasserstein(heatmap, mouse_seg)
        rmse = compute_rmse(heatmap, mouse_seg)

        all_ious.append(iou)
        all_emds.append(emd)
        all_rmses.append(rmse)

    return np.mean(all_ious), np.mean(all_emds), np.mean(all_rmses)

# =========================================================
# DATASET EVAL
# =========================================================

def evaluate_dataset_nested(slide_root, data_root):

    all_ious = []
    all_emds = []
    all_rmses = []

    for root, _, files in os.walk(slide_root):

        for file in files:

            if not file.endswith(".png"):
                continue

            slide_path = os.path.join(root, file)
            relative_folder = os.path.relpath(root, slide_root)
            slide_id = file.replace(".png", "")

            asr_path = os.path.join(
                data_root,
                relative_folder,
                slide_id + "_spoken.csv"
            )

            mouse_path = os.path.join(
                data_root,
                relative_folder,
                slide_id + "_trace.csv"
            )

            if not os.path.exists(asr_path) or not os.path.exists(mouse_path):
                continue

            print("Processing:", relative_folder, slide_id)

            iou, emd, rmse = evaluate_clip_slide(
                slide_path, asr_path, mouse_path
            )

            all_ious.append(iou)
            all_emds.append(emd)
            all_rmses.append(rmse)

    print("================================")
    print("Slides processed:", len(all_ious))
    print("Average CLIP IoU:", np.mean(all_ious))
    print("Average CLIP Wasserstein:", np.mean(all_emds))
    print("Average CLIP RMSE:", np.mean(all_rmses))

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    slide_dir = r"Figures/anat-2/ALRJCeVT0fQ"
    data_dir  = r"mlpdataset/data_oct/anat-2/unordered/ALRJCeVT0fQ"

    evaluate_dataset_nested(slide_dir, data_dir)