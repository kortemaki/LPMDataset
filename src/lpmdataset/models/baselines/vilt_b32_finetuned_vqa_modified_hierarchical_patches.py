from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from lpmdataset.data_models import Slide


def ocr_bbs_to_xyxy(height: int, width: int, df: pd.DataFrame) -> np.array:
    """
    Transform the left, top, width, height format for bounding boxes
    into [x_min, y_min, x_max, y_max] in [0, 1].

    """
    return np.stack(
        (
            np.clip(df['left'], 0, width) / width,
            np.clip(df['top'], 0, height) / height,
            np.clip(df['left'] + df['width'], 0, width) / width,
            np.clip(df['top'] + df['height'], 0, height) / height,
        ),
        axis=1,
    )


def patch_boxes_xyxy(height, width, kernel=(32,32), stride=(32,32), inclusive=False) -> np.array:
    """
    Returns an array of shape (n_patches, 4) with columns [x_min, y_min, x_max, y_max].
    Coordinates are in pixels where x is horizontal (cols) and y is vertical (rows).

    """
    kh, kw = kernel
    sh, sw = stride

    rows = np.arange(0, max(1, height - kh + 1), sh)
    cols = np.arange(0, max(1, width  - kw + 1), sw)
    R, C = np.meshgrid(rows, cols, indexing='ij')   # shapes (n_rows, n_cols)
    toplefts = np.stack((R.ravel(), C.ravel()), axis=1)  # (N,2) as (y, x)

    y = toplefts[:, 0]
    x = toplefts[:, 1]

    if inclusive:
        x_max = x + kw - 1
        y_max = y + kh - 1
    else:
        x_max = x + kw
        y_max = y + kh

    # Clip to image bounds: x in [0, width], y in [0, height]
    x = np.clip(x, 0, width) / width
    x_max = np.clip(x_max, 0, width) / width
    y = np.clip(y, 0, height) / height
    y_max = np.clip(y_max, 0, height) / height

    return np.stack((x, y, x_max, y_max), axis=1)


def generate_candidate_space(
    word_embeddings,   # Shape: (N_words, 768)
    word_bboxes,       # Shape: (N_words, 4) -> [x_min, y_min, x_max, y_max] in [0, 1]
    patch_embeddings,  # Shape: (N_patches, 768)
    patch_bboxes,      # Shape: (N_patches, 4) -> [x_min, y_min, x_max, y_max] in [0, 1]
    grid_size=32,
    embed_dim=768
):
    """
    Produces a spatial candidate matrix by area-weighting word and patch embeddings.
    """
    # 1. Initialize the grid and a weight accumulator
    # shape: (1024, 768)
    candidate_space = np.zeros((grid_size**2, embed_dim))
    weight_sums = np.zeros((grid_size**2, 1))

    # Define grid cell dimensions
    cell_dim = 1.0 / grid_size


    # 2. Process Word and Patch Embeddings via Area-Weighting
    for embs, emb_bboxes in ((word_embeddings, word_bboxes), (patch_embeddings, patch_bboxes)):
        for i in range(min(len(embs), len(emb_bboxes))):
            emb = embs[i]
            x_min, y_min, x_max, y_max = emb_bboxes[i]

            # Determine which grid indices the bounding box potentially touches
            col_start = int(np.floor(x_min * grid_size))
            col_end = int(np.ceil(x_max * grid_size))
            row_start = int(np.floor(y_min * grid_size))
            row_end = int(np.ceil(y_max * grid_size))

            # Iterate only over affected cells
            for r in range(max(0, row_start), min(grid_size, row_end)):
                for c in range(max(0, col_start), min(grid_size, col_end)):
                    # Calculate coordinates of the current grid cell
                    cell_x_min, cell_y_min = c * cell_dim, r * cell_dim
                    cell_x_max, cell_y_max = (c + 1) * cell_dim, (r + 1) * cell_dim

                    # Calculate Intersection Area
                    inter_x_min = max(x_min, cell_x_min)
                    inter_y_min = max(y_min, cell_y_min)
                    inter_x_max = min(x_max, cell_x_max)
                    inter_y_max = min(y_max, cell_y_max)

                    inter_w = max(0, inter_x_max - inter_x_min)
                    inter_h = max(0, inter_y_max - inter_y_min)
                    intersection_area = inter_w * inter_h

                    # Normalize area relative to the grid cell's total area (cell_dim^2)
                    # This gives the "coverage" weight
                    weight = intersection_area / (cell_dim ** 2)

                    if weight > 0:
                        idx = r * grid_size + c
                        emb = np.asarray(emb, dtype=np.float32).flatten()
                        candidate_space[idx] += emb * weight
                        weight_sums[idx] += weight

    # 3. Final Weighted Average
    # Avoid division by zero for empty cells (though patch_embeddings ensure > 0)
    candidate_space /= np.maximum(weight_sums, 1e-8)

    return candidate_space # Final shape (1024, 768)


def get_best_regions(index, queries) -> np.ndarray:
    """return the 2D index of the closest region to each query."""
    index_norm = index / np.linalg.norm(index, axis=1, keepdims=True)
    query_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    sims = query_norm.dot(index_norm.T)    # (N,)
    return np.argmax(sims, axis=1)


class ModifiedVilt(nn.Module):
    def __init__(self, base_model: nn.Module, train: bool=False) -> None:
        super(ModifiedVilt, self).__init__()
        self.vilt = base_model.vilt
        if not train:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, pixel_values, pixel_mask=None, token_type_ids=None):
        outputs = self.vilt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,
        )
        return outputs


# ---------- grid / composite constants ------------------------------------
GRID_COLS        = 12
GRID_SLIDE_ROWS  = 8
GRID_BANNER_ROWS = 4
GRID_TOTAL_ROWS  = GRID_SLIDE_ROWS + GRID_BANNER_ROWS  # 12
PATCH_PX         = 32
COMPOSITE_SIZE   = GRID_COLS * PATCH_PX                 # 384


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ModifiedVilt(ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")).to(device)


def build_stacked_image_and_mapping(image: Image.Image, slide: Slide):
    """
    Build a 384x384 composite: slide on top (12x8 patch-units) and a
    figure banner on the bottom (12x4 units).

    Figures are cropped from the original image, resized to their banner
    slot, and pasted left-to-right.  The composite is exactly 384x384 so
    that ViLT's 32x32 patches produce a deterministic 12x12 grid.

    Returns
    -------
    stacked : PIL.Image  (384x384 RGB)
    mapping_info : dict
        ``patch_origins`` maps every ``(row, col)`` in the 12x12 logical
        grid to ``(x_min, y_min, x_max, y_max)`` in the original slide's
        pixel coordinates, or ``None`` for blank banner cells.
    """
    orig_w, orig_h = image.size

    slide_h  = GRID_SLIDE_ROWS  * PATCH_PX   # 256
    banner_h = GRID_BANNER_ROWS * PATCH_PX    # 128

    # --- resize slide into the top portion --------------------------------
    slide_resized = image.resize((COMPOSITE_SIZE, slide_h), Image.LANCZOS)

    # --- build the figure banner ------------------------------------------
    banner = Image.new("RGB", (COMPOSITE_SIZE, banner_h), (0, 0, 0))

    figures_df = slide.figures.copy()
    n_figs = len(figures_df)
    if n_figs > 12:
        figures_df = figures_df.iloc[:12]
        n_figs = 12

    # Each entry: (df_row_idx, x_px, y_px, w_px, h_px) in banner pixel space
    fig_placements = []

    if n_figs == 1:
        # 12 x 4 units – full banner
        fig_placements.append((0, 0, 0, 12 * PATCH_PX, 4 * PATCH_PX))
    elif n_figs == 2:
        # 6 x 4 each
        for i in range(2):
            fig_placements.append(
                (i, i * 6 * PATCH_PX, 0, 6 * PATCH_PX, 4 * PATCH_PX))
    elif n_figs == 3:
        # 4 x 4 each
        for i in range(3):
            fig_placements.append(
                (i, i * 4 * PATCH_PX, 0, 4 * PATCH_PX, 4 * PATCH_PX))
    elif n_figs == 4:
        # 3 x 4 each
        for i in range(4):
            fig_placements.append(
                (i, i * 3 * PATCH_PX, 0, 3 * PATCH_PX, 4 * PATCH_PX))
    elif 5 <= n_figs <= 6:
        # 2 x 4 each
        for i in range(n_figs):
            fig_placements.append(
                (i, i * 2 * PATCH_PX, 0, 2 * PATCH_PX, 4 * PATCH_PX))
    elif 7 <= n_figs <= 12:
        # Two rows of 2-unit height; each figure 2 x 2 units (6 per row)
        for i in range(n_figs):
            row_i = 0 if i < 6 else 1
            col_i = i if i < 6 else (i - 6)
            fig_placements.append((
                i,
                col_i * 2 * PATCH_PX,
                row_i * 2 * PATCH_PX,
                2 * PATCH_PX,
                2 * PATCH_PX,
            ))

    # Crop, resize, and paste each figure ----------------------------------
    fig_original_bboxes = {}  # fig_idx -> (left, top, right, bottom) orig px

    for fig_idx, x_off, y_off, fw, fh in fig_placements:
        r = figures_df.iloc[fig_idx]
        left   = int(np.clip(r['left'], 0, orig_w))
        top_   = int(np.clip(r['top'],  0, orig_h))
        right  = int(np.clip(r['left'] + r['width'],  0, orig_w))
        bottom = int(np.clip(r['top']  + r['height'], 0, orig_h))

        fig_original_bboxes[fig_idx] = (left, top_, right, bottom)

        if right > left and bottom > top_:
            crop = image.crop((left, top_, right, bottom))
            crop = crop.resize((fw, fh), Image.LANCZOS)
            banner.paste(crop, (x_off, y_off))

    # --- compose 384x384 stacked image ------------------------------------
    stacked = Image.new("RGB", (COMPOSITE_SIZE, COMPOSITE_SIZE))
    stacked.paste(slide_resized, (0, 0))
    stacked.paste(banner, (0, slide_h))

    # --- build 12x12 logical-patch -> original-pixel mapping --------------
    patch_origins = {}  # (row, col) -> (x_min, y_min, x_max, y_max) | None

    # Slide rows 0 .. GRID_SLIDE_ROWS-1
    for lr in range(GRID_SLIDE_ROWS):
        for lc in range(GRID_COLS):
            patch_origins[(lr, lc)] = (
                lc       / GRID_COLS       * orig_w,
                lr       / GRID_SLIDE_ROWS * orig_h,
                (lc + 1) / GRID_COLS       * orig_w,
                (lr + 1) / GRID_SLIDE_ROWS * orig_h,
            )

    # Banner rows GRID_SLIDE_ROWS .. GRID_TOTAL_ROWS-1
    for lr_b in range(GRID_BANNER_ROWS):
        lr = GRID_SLIDE_ROWS + lr_b
        for lc in range(GRID_COLS):
            px_x0 = lc       * PATCH_PX
            px_y0 = lr_b     * PATCH_PX
            px_x1 = (lc + 1) * PATCH_PX
            px_y1 = (lr_b + 1) * PATCH_PX

            found = False
            for fig_idx, fx, fy, fw, fh in fig_placements:
                if fx < px_x1 and px_x0 < fx + fw and fy < px_y1 and px_y0 < fy + fh:
                    left, top_, right, bottom = fig_original_bboxes[fig_idx]
                    fig_w = right - left
                    fig_h = bottom - top_

                    ox0 = max(px_x0, fx)
                    oy0 = max(px_y0, fy)
                    ox1 = min(px_x1, fx + fw)
                    oy1 = min(px_y1, fy + fh)

                    frac_x0 = (ox0 - fx) / fw
                    frac_y0 = (oy0 - fy) / fh
                    frac_x1 = (ox1 - fx) / fw
                    frac_y1 = (oy1 - fy) / fh

                    patch_origins[(lr, lc)] = (
                        left + frac_x0 * fig_w,
                        top_ + frac_y0 * fig_h,
                        left + frac_x1 * fig_w,
                        top_ + frac_y1 * fig_h,
                    )
                    found = True
                    break

            if not found:
                patch_origins[(lr, lc)] = None

    mapping_info = dict(
        orig_w=orig_w,
        orig_h=orig_h,
        fig_placements=fig_placements,
        fig_original_bboxes=fig_original_bboxes,
        patch_origins=patch_origins,
    )
    return stacked, mapping_info


def project_heatmap_to_original(heatmap_np, mapping_info, output_shape=None):
    """
    Re-project a heatmap produced on the stacked image back into the
    original slide's coordinate space.

    Slide-portion patches are rescaled proportionally.  Banner-portion
    patches are mapped through their figure's original bounding box.
    Blank banner patches are discarded.
    """
    grid_h, grid_w = heatmap_np.shape
    if output_shape is None:
        output_shape = (grid_h, grid_w)
    out_rows, out_cols = output_shape

    orig_w          = mapping_info['orig_w']
    orig_h          = mapping_info['orig_h']
    fig_placements  = mapping_info['fig_placements']
    fig_original_bboxes = mapping_info['fig_original_bboxes']

    slide_frac = GRID_SLIDE_ROWS / GRID_TOTAL_ROWS   # 8/12

    projected = np.zeros(output_shape, dtype=np.float64)

    for r in range(grid_h):
        for c in range(grid_w):
            w = float(heatmap_np[r, c])
            if w <= 0:
                continue

            # Centre of this model-patch in composite normalised coords
            y_frac = (r + 0.5) / grid_h
            x_frac = (c + 0.5) / grid_w

            if y_frac <= slide_frac:
                # Slide region – proportional mapping
                x_orig_n = x_frac
                y_orig_n = y_frac / slide_frac
            else:
                # Banner region – map through figure bbox
                y_banner_px = ((y_frac - slide_frac)
                               / (1 - slide_frac)
                               * GRID_BANNER_ROWS * PATCH_PX)
                x_banner_px = x_frac * COMPOSITE_SIZE

                hit = False
                for fig_idx, fx, fy, fw, fh in fig_placements:
                    if (fx <= x_banner_px < fx + fw
                            and fy <= y_banner_px < fy + fh):
                        left, top_, right, bottom = \
                            fig_original_bboxes[fig_idx]
                        x_orig = (left
                                  + (x_banner_px - fx) / fw
                                  * (right - left))
                        y_orig = (top_
                                  + (y_banner_px - fy) / fh
                                  * (bottom - top_))
                        x_orig_n = x_orig / orig_w
                        y_orig_n = y_orig / orig_h
                        hit = True
                        break
                if not hit:
                    continue   # blank banner area – discard

            out_r = min(int(y_orig_n * out_rows), out_rows - 1)
            out_c = min(int(x_orig_n * out_cols), out_cols - 1)
            projected[out_r, out_c] += w

    return projected


def predict(slide: Slide):
    # Get image
    image = Image.open(slide.png).convert("RGB")

    # Build 384x384 stacked image (slide + figure banner) and patch mapping
    stacked_image, mapping_info = build_stacked_image_and_mapping(image, slide)

    results = []

    # NOTE: You will need to adjust this loop depending on how your
    # slide.asr_text object stores time boundaries.
    # I am assuming a method that yields both the bounds and the text.
    for sent, tbounds in slide.asr_text.to_sentences():

        # Encode stacked image and sentence
        encoding_slide_asr = processor(
            stacked_image,
            sent,
            return_tensors="pt",
            truncation=True,
            max_length=40,
            padding="max_length",
        )
        encoding_slide_asr = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in encoding_slide_asr.items()}
        outputs = model(**encoding_slide_asr)

        # 1. Map out the sequence length
        input_ids = encoding_slide_asr["input_ids"]
        num_text_tokens = 40 #input_ids.shape[1]

        # 2. Map out the image patch grid (on the 384x384 composite)
        # 384 / 32 = 12 -> deterministic 12x12 grid
        _, _, H, W = encoding_slide_asr["pixel_values"].shape
        grid_h, grid_w = H // 32, W // 32
        num_patches = grid_h * grid_w

        # 3. Extract the raw attention from the last 2 layers
        # Shape: (seq_len, seq_len)
        avg_attn = torch.stack(outputs.attentions[-2:]).mean(dim=0).mean(dim=1)[0]

        # 4. Define our Source (Text) and Target (Image Patches)
        # Text tokens occupy indices 0 to (num_text_tokens - 1).
        # We slice from 1 to -1 to ignore the [CLS] and [SEP] special tokens.
        text_indices = np.arange(1, num_text_tokens - 1)

        # Image patches sit immediately after the text tokens
        patch_start = num_text_tokens
        patch_end = num_text_tokens + num_patches

        # 5. Slice the attention map (From Text -> To Patches)
        # Shape: (len(text_indices), num_patches)
        text_to_patch_attn = avg_attn[text_indices, patch_start:patch_end]

        # 6. Collapse into a heatmap over the stacked-image grid
        heatmap_stacked = text_to_patch_attn.mean(dim=0).view(grid_h, grid_w)
        heatmap_stacked_np = heatmap_stacked.detach().cpu().numpy()

        # 7. Project heatmap back onto original slide coordinates
        heatmap_np = project_heatmap_to_original(
            heatmap_stacked_np, mapping_info)

        # 8. Normalize
        heatmap_np = heatmap_np / (heatmap_np.sum() + 1e-8)

        results.append((tbounds, sent, heatmap_np))

    return results


"""
def predict(slide: Slide) -> list[tuple[int,int]]:
    # get image and OCR data
    image = Image.open(slide.png).convert("RGB")
    encoding_ocr = processor(None, slide.ocr_text.to_string(), return_tensors="pt")['input_ids'][0][1:-1]
    embedding_ocr = model.vilt.embeddings.text_embeddings.word_embeddings(encoding_ocr)  # shape (batch_size, seq_length, hidden_size)
    encoding_ocr_bbs = np.array([
        tok_bb
        for tok_bb, tok_word in zip(ocr_bbs_to_xyxy(height=image.height, width=image.width, df=slide.ocr_text.bbs), slide.ocr_text.tokens)
        for tok in processor.tokenizer.tokenize(tok_word)
    ])

    best_regions_raw = []

    for sent, _ in slide.asr_text.to_sentences():
        # encode slide
        encoding_slide_asr = processor(image, sent, return_tensors="pt")
        outputs = model(**encoding_slide_asr)

        # prepare candidate space
        embedding_patches = model.vilt.embeddings.patch_embeddings(encoding_slide_asr['pixel_values'])  # shape (batch_size, hidden_size, height / 32, width / 32)
        patch_bbs = patch_boxes_xyxy(image.height, image.width)
        regions = generate_candidate_space(
            embedding_ocr,
            encoding_ocr_bbs,
            embedding_patches.reshape(embedding_patches.shape[1], -1).T,
            patch_bbs,
        )

        # decode best regions
        i = 1
        queries = []
        for query_word in sent.split():
            query_tokens = processor.tokenizer.tokenize(query_word)
            k = len(query_tokens)
            queries.append(outputs[0, i:i + k, :].mean(axis=0))  # shape (hidden_size,)
            i += k
        best_regions_raw.append(get_best_regions(regions, queries))

    best_regions = np.concatenate(best_regions_raw, axis=0)
    return list(
        zip(
            (best_regions % 32).tolist(),  # x indices of each prediction
            (best_regions // 32).tolist(),  # y indices of each prediction
        )
    )
"""
