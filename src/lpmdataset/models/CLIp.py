import torch
import open_clip
import numpy as np
import pandas as pd
import os
from PIL import Image

# =========================================================
# DEVICE
# =========================================================
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
# BUILD LECTURE MAP  (FIXED)
#
# The mapping problem has two layers:
#
# 1. data_oct folder: <channel>/<playlist>/<lecture_num>/
#    lecture_num is the YouTube PLAYLIST POSITION (1-based), with gaps
#    for videos that exist in the playlist but had no annotations collected.
#    e.g. speaking lecture 30 exists in data_oct but raw_video_links only
#         has 24 speaking rows because 6 playlist videos were never annotated.
#
# 2. figures folder: <channel>/<youtube_id>/
#    No playlist subfolder, no lecture number — just the bare YouTube ID.
#
# The bridge:
#   - raw_video_links rows filtered by channel, kept in file order,
#     are in the same order as the YouTube playlist.
#   - figure_annotations tells us exactly which lecture numbers appear
#     in the dataset for this channel (sorted).
#   - Zipping these two gives the correct lecture_num -> youtube_id map.
#
# Returns: { "01": "_Awekr6-ilg", "03": "0gf3IJFTxEA", ... }
# =========================================================
def build_lecture_map(raw_video_links_csv, figure_annotations_csv, channel):
    df_videos = pd.read_csv(raw_video_links_csv)
    df_ann    = pd.read_csv(figure_annotations_csv)

    # Filter to this channel, preserving file order (= playlist order)
    channel_videos = df_videos[df_videos["speaker"] == channel].reset_index(drop=True)
    if channel_videos.empty:
        raise ValueError(f"No rows for channel '{channel}' in raw_video_links.csv")

    # Extract sorted lecture numbers for this channel from figure_annotations
    parts = df_ann["Input.save_dir"].str.split("/", expand=True)
    df_ann["_channel"]     = parts[1]
    df_ann["_lecture_num"] = parts[3]

    channel_lec_nums = sorted(
        df_ann[df_ann["_channel"] == channel]["_lecture_num"]
        .dropna()
        .astype(int)
        .unique()
    )

    if len(channel_lec_nums) != len(channel_videos):
        print(
            f"  ⚠️  '{channel}': {len(channel_videos)} videos in raw_video_links "
            f"vs {len(channel_lec_nums)} lecture nums in figure_annotations. "
            f"Using first {min(len(channel_videos), len(channel_lec_nums))} entries."
        )

    n = min(len(channel_videos), len(channel_lec_nums))
    lecture_map = {}
    for i in range(n):
        lec_key    = f"{channel_lec_nums[i]:02d}"
        youtube_id = str(channel_videos.iloc[i]["video_id"]).replace(".mp4", "").strip()
        lecture_map[lec_key] = youtube_id

    print(f"✅ Lecture map for '{channel}' ({len(lecture_map)} entries):")
    for lec, vid in lecture_map.items():
        print(f"   {lec} -> {vid}")

    return lecture_map


# =========================================================
# SAFE CSV
# =========================================================
def safe_read_csv(path):
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception:
        print(f"  ⚠️  Bad CSV: {path}")
        return pd.DataFrame()


# =========================================================
# ASR SEGMENTS
# =========================================================
def build_segments(asr_path, window=2.0):
    df = safe_read_csv(asr_path)
    if df.empty:
        return []

    df["Start"] = df["Start"].clip(lower=0)
    segments = []
    t = df["Start"].min()
    end_time = df["End"].max()

    while t < end_time:
        chunk = df[(df["Start"] >= t) & (df["End"] <= t + window)]
        if len(chunk) > 0:
            text = " ".join(chunk["Word"].astype(str).tolist())
            segments.append({"start": t, "end": t + window, "text": text})
        t += window

    return segments


# =========================================================
# LOAD & NORMALIZE MOUSE
# =========================================================
def load_mouse(mouse_path):
    df = safe_read_csv(mouse_path)
    if df.empty:
        return df
    df[["x", "y"]] = df["coord"].str.extract(r"\((\d+),\s*(\d+)\)")
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    return df


def normalize_mouse(df, screen_w=854, screen_h=480):
    if df.empty:
        return df
    df["x"] /= screen_w
    df["y"] /= screen_h
    return df


# =========================================================
# PATCH EXTRACTION & ENCODING
# =========================================================
def extract_patches(image, grid_size=16):
    W, H = image.size
    patch_w = W // grid_size
    patch_h = H // grid_size
    patches = []
    for r in range(grid_size):
        for c in range(grid_size):
            left = c * patch_w
            top  = r * patch_h
            patches.append(image.crop((left, top, left + patch_w, top + patch_h)))
    return patches


def encode_patches(patches):
    processed = [preprocess(p) for p in patches]
    image_inputs = torch.stack(processed).to(device)
    with torch.no_grad():
        feats = model.encode_image(image_inputs)
    return feats / feats.norm(dim=-1, keepdim=True)


# =========================================================
# TEXT -> HEATMAP -> POINT
# =========================================================
def text_to_heatmap(text, image_features, grid_size=16):
    if text.strip() == "":
        return np.ones((grid_size, grid_size)) / (grid_size ** 2)

    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    sim = (image_features @ text_features.T).squeeze()
    heatmap = torch.softmax(sim, dim=0)
    return heatmap.reshape(grid_size, grid_size).cpu().numpy()


def heatmap_to_point(heatmap):
    grid_size = heatmap.shape[0]
    xs = np.linspace(0, 1, grid_size)
    ys = np.linspace(0, 1, grid_size)
    exp_x, exp_y = 0.0, 0.0
    for i in range(grid_size):
        for j in range(grid_size):
            exp_x += xs[j] * heatmap[i, j]
            exp_y += ys[i] * heatmap[i, j]
    return exp_x, exp_y


# =========================================================
# GET SLIDE PATH  (FIXED)
#
# data_oct: <data_root>/<channel>/<playlist>/<lecture_num>/<slide>_trace.csv
# figures:  <slide_root>/<channel>/<youtube_id>/<slide>.jpg
#
# No playlist in figures path. No lecture number either.
# Just resolve: lecture_num -> youtube_id via lecture_map.
# =========================================================
def get_slide_path(slide_root, channel, lecture_num, slide_id, lecture_map):
    lecture_key = f"{int(lecture_num):02d}"

    if lecture_key not in lecture_map:
        return None

    youtube_id = lecture_map[lecture_key]
    slide_dir  = os.path.join(slide_root, channel, youtube_id)

    for ext in [".jpg", ".png"]:
        path = os.path.join(slide_dir, slide_id + ext)
        if os.path.exists(path):
            return path

    return None


# =========================================================
# PROCESS ONE SLIDE
# =========================================================
def process_slide(slide_path, asr_path, mouse_path, save_path):
    image = Image.open(slide_path).convert("RGB")
    patches = extract_patches(image)
    image_features = encode_patches(patches)

    segments = build_segments(asr_path)
    mouse_df = normalize_mouse(load_mouse(mouse_path))

    if mouse_df.empty:
        return

    rows = []
    for seg in segments:
        mouse_seg = mouse_df[
            (mouse_df["time"] >= seg["start"]) &
            (mouse_df["time"] <= seg["end"])
        ]
        if len(mouse_seg) == 0:
            continue

        heatmap = text_to_heatmap(seg["text"], image_features)
        pred_x, pred_y = heatmap_to_point(heatmap)

        for _, r in mouse_seg.iterrows():
            rows.append({
                "time":   r["time"],
                "pred_x": pred_x,
                "pred_y": pred_y,
                "gold_x": r["x"],
                "gold_y": r["y"],
            })

    if not rows:
        return

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_out.to_csv(save_path, index=False)
    print(f"    ✅ Saved {len(rows)} rows -> {save_path}")


# =========================================================
# MAIN PIPELINE  (FIXED)
#
# Walks: data_root/<channel>/<playlist>/<lecture_num>/
# Finds _trace.csv files, resolves slide images via lecture_map.
# =========================================================
def run_pipeline(slide_root, data_root, test_channels,
                 raw_video_links_csv, figure_annotations_csv):

    for channel in test_channels:
        print(f"\n{'='*60}")
        print(f"CHANNEL: {channel}")
        print(f"{'='*60}")

        lecture_map = build_lecture_map(
            raw_video_links_csv, figure_annotations_csv, channel
        )

        channel_dir = os.path.join(data_root, channel)
        if not os.path.isdir(channel_dir):
            print(f"  ⚠️  Directory not found: {channel_dir}")
            continue

        # Walk: channel_dir/<playlist>/<lecture_num>/
        for playlist in sorted(os.listdir(channel_dir)):
            playlist_dir = os.path.join(channel_dir, playlist)
            if not os.path.isdir(playlist_dir):
                continue

            for lecture_num in sorted(os.listdir(playlist_dir)):
                lecture_dir = os.path.join(playlist_dir, lecture_num)
                if not os.path.isdir(lecture_dir):
                    continue

                trace_files = [f for f in os.listdir(lecture_dir)
                               if f.endswith("_trace.csv")]
                if not trace_files:
                    continue

                print(f"\n  [{channel}/{playlist}/{lecture_num}]")

                for trace_file in sorted(trace_files):
                    slide_id = trace_file.replace("_trace.csv", "")

                    slide_path = get_slide_path(
                        slide_root, channel, lecture_num, slide_id, lecture_map
                    )
                    if slide_path is None:
                        print(f"    ⚠️  No slide: lecture={lecture_num} {slide_id}")
                        continue

                    asr_path   = os.path.join(lecture_dir, slide_id + "_spoken.csv")
                    mouse_path = os.path.join(lecture_dir, trace_file)

                    if not os.path.exists(asr_path):
                        print(f"    ⚠️  No ASR file for {slide_id}")
                        continue

                    save_path = os.path.join(
                        "results", "clip", channel, lecture_num, slide_id + ".csv"
                    )
                    print(f"    Processing {slide_id} ...")
                    try:
                        process_slide(slide_path, asr_path, mouse_path, save_path)
                    except Exception as e:
                        print(f"    ❌ Failed: {slide_id} | {e}")


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":

    SLIDE_ROOT = r"C:/Users/saumy/figures/figures"
    DATA_ROOT  = r"C:/Users/saumy/LPMDatasetRepo/LPMDataset/mlpdataset/data_oct"

    RAW_VIDEO_LINKS_CSV    = os.path.join(DATA_ROOT, "raw_video_links.csv")
    FIGURE_ANNOTATIONS_CSV = os.path.join(DATA_ROOT, "figure_annotations.csv")

    # Must match 'speaker' column values in raw_video_links.csv
    TEST_CHANNELS = ["ml-1", "speaking"]

    run_pipeline(
        slide_root             = SLIDE_ROOT,
        data_root              = DATA_ROOT,
        test_channels          = TEST_CHANNELS,
        raw_video_links_csv    = RAW_VIDEO_LINKS_CSV,
        figure_annotations_csv = FIGURE_ANNOTATIONS_CSV,
    )