from transformers import LayoutLMv3Processor, LayoutLMv3Model
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from lpmdataset.data_models import Slide


def create_layoutlmv3_input_sequence(
    pruned_ocr_df,
    asr_text,
    tokenizer,
    max_seq_len=512
):
    """
    Constructs input_ids and bboxes for the sequence:
    <s> [OCR] </s> </s> [ASR] </s> [PAD]
    """
    # 1. Define null constants
    null_box = [0, 0, 0, 0]
    bos_id = tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id

    # 2. Build OCR Sequence
    ocr_ids = []
    ocr_boxes = []
    for _, row in pruned_ocr_df.iterrows():
        # Ensure a leading space so tokens are consistent
        # LayoutLMv3 tokenizer: 'word' vs ' word' have different IDs
        w_ids = tokenizer._tokenizer.encode(" " + str(row['text']), add_special_tokens=False).ids
        w_box = row['scaled_box'] # From our previous scaling logic

        ocr_ids.extend(w_ids)
        ocr_boxes.extend([w_box] * len(w_ids))

    # 3. Build ASR Sequence
    asr_ids = tokenizer._tokenizer.encode(" " + asr_text, add_special_tokens=False).ids
    # ASR tokens always get null boxes since they are not physically there
    asr_boxes = [null_box] * len(asr_ids)

    # 4. Concatenate with Special Tokens
    # LayoutLMv3 usually uses <s> Text </s> </s> Text </s> for pair sequences
    final_input_ids = [bos_id] + ocr_ids + [sep_id, sep_id] + asr_ids + [sep_id]
    final_bboxes = [null_box] + ocr_boxes + [null_box, null_box] + asr_boxes + [null_box]

    # 5. Padding
    # Important: padding_idx for LayoutLMv3 is usually 1
    pad_len = max_seq_len - len(final_input_ids)
    if pad_len > 0:
        final_input_ids += [tokenizer.pad_token_id] * pad_len
        final_bboxes += [null_box] * pad_len
    else:
        # If ASR is too long, we truncate from the end of the ASR (before the final </s>)
        final_input_ids = final_input_ids[:max_seq_len]
        final_bboxes = final_bboxes[:max_seq_len]

    # 6. Track ASR indices for Attention Extraction
    # We need to know where the ASR tokens are in the 512 sequence
    # Offset by 196+1 because the model prepends image patches internally
    patch_offset = 196
    # ASR starts after <s> (1), OCR (len), and </s> </s> (2)
    asr_start_idx = 1 + patch_offset + 1 + len(ocr_ids) + 2
    asr_end_idx = asr_start_idx + len(asr_ids)

    # 7. Set attention mask to have ASR only look at slide and past ASR
    # Create 1D mask: 1 for content, 0 for padding
    # Total length 512 (text portion only)
    attention_mask = torch.zeros((1, max_seq_len))
    attention_mask[0][:1 + len(ocr_ids) + 2 + len(asr_ids) + 1] = 1

    return {
        "input_ids": torch.tensor([final_input_ids], dtype=torch.long),
        "bbox": torch.tensor([final_bboxes], dtype=torch.long),
        "asr_range": (asr_start_idx, asr_end_idx),
        "attention_mask": attention_mask,
    }


def get_attention_heatmap(outputs, asr_token_indices):
    """
    asr_token_indices: The indices in the 512-length sequence where ASR sits.
    Example: If OCR is 1-200, ASR starts at 202 (after <s><s>).
    """
    # Use the last 2 layers for high-level semantic grounding
    # Mean across heads: (709, 709)
    avg_attn = torch.stack(outputs.attentions[-2:]).mean(dim=0).mean(dim=1)[0]

    # We want: From [ASR Tokens] -> To [Image Patches (0:196)]
    # Shape: (len(asr_tokens), 196)
    asr_to_patch_attn = avg_attn[np.arange(*asr_token_indices), -197:-1]

    # Average across all tokens in the ASR sentence to get one 'Attention Map'
    # Shape: (196,) -> Resize to (14, 14) grid
    heatmap = asr_to_patch_attn.mean(dim=0).view(14, 14)

    # Return raw (un-normalised) heatmap; normalisation happens after
    # projection back to original-slide coordinates.
    return heatmap.detach().cpu().numpy()


def measure_grounding_score(word, tokenizer, encoding_slide, outputs):
    """
    Measures the specific attention 'handshake' between a word spoken (ASR)
    and that same word written on the slide (OCR).
    """
    # 1. Tokenize the word as it would appear in the sequence (with prefix space)
    target_ids = tokenizer._tokenizer.encode(" " + word, add_special_tokens=False).ids
    input_ids = encoding_slide['input_ids'][0].tolist()

    # 2. Find all occurrences of the token sequence in input_ids
    # We need to distinguish between the OCR section and the ASR section
    def find_all(full, sub):
        res = []
        for i in range(len(full) - len(sub) + 1):
            if full[i : i + len(sub)] == sub:
                res.append(list(range(i, i + len(sub))))
        return res

    occurrences = find_all(input_ids, target_ids)

    if len(occurrences) < 2:
        return f"Word '{word}' not found in both sections. Occurrences: {len(occurrences)}"

    # 3. Identify which occurrence is OCR and which is ASR
    # Based on our 709 structure: Patches(0-195), CLS(196), Text(197-708)
    # The first text occurrence is almost certainly OCR (since OCR comes first)
    ocr_indices = occurrences[0] # Shift for Patch+CLS offset
    asr_indices = occurrences[-1]

    # 4. Extract Attention Matrix (Average of last 2 layers, average of heads)
    # Shape: (709, 709)
    attn_matrix = torch.stack(outputs.attentions[-2:]).mean(dim=0).mean(dim=1)[0]

    # 5. Calculate Cross-Attention
    # We want: From ASR (Rows) -> To OCR (Cols)
    # We take the mean of the sub-matrix defined by these indices
    grounding_score = attn_matrix[asr_indices][:, ocr_indices].mean().item()

    # Baseline check: What is the average attention this ASR word gives to ANYTHING?
    global_average = attn_matrix[asr_indices, :].mean().item()

    # also check visual score
    visual_grounding = attn_matrix[asr_indices, -197:-1].mean().item()

    return {
        "word": word,
        "grounding_score": grounding_score,
        "relative_strength": grounding_score / (global_average + 1e-8),
        "visual_grounding": visual_grounding,
        "ocr_indices": ocr_indices,
        "asr_indices": asr_indices
    }


# ---------- grid / composite constants ------------------------------------
GRID_COLS        = 14
GRID_SLIDE_ROWS  = 10
GRID_BANNER_ROWS = 4
GRID_TOTAL_ROWS  = GRID_SLIDE_ROWS + GRID_BANNER_ROWS  # 14
PATCH_PX         = 16
COMPOSITE_SIZE   = GRID_COLS * PATCH_PX                 # 224


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3Model.from_pretrained("gordonlim/layoutlmv3-base-finetuned-rvlcdip").to(device)


def build_stacked_image_and_mapping(image: Image.Image, slide: Slide):
    """
    Build a 224x224 composite: slide on top (14x10 patch-units) and a
    figure banner on the bottom (14x4 units).

    Figures are cropped from the original image, resized to their banner
    slot, and pasted left-to-right.  The composite is exactly 224x224 so
    that LayoutLMv3's 16x16 patches produce a deterministic 14x14 grid.

    Returns
    -------
    stacked : PIL.Image  (224x224 RGB)
    mapping_info : dict
        ``patch_origins`` maps every ``(row, col)`` in the 14x14 logical
        grid to ``(x_min, y_min, x_max, y_max)`` in the original slide's
        pixel coordinates, or ``None`` for blank banner cells.
    """
    orig_w, orig_h = image.size

    slide_h  = GRID_SLIDE_ROWS  * PATCH_PX   # 160
    banner_h = GRID_BANNER_ROWS * PATCH_PX    # 64

    # --- resize slide into the top portion --------------------------------
    slide_resized = image.resize((COMPOSITE_SIZE, slide_h), Image.LANCZOS)

    # --- build the figure banner ------------------------------------------
    banner = Image.new("RGB", (COMPOSITE_SIZE, banner_h), (0, 0, 0))

    figures_df = slide.figures.copy()
    n_figs = len(figures_df)
    if n_figs > 14:
        figures_df = figures_df.iloc[:14]
        n_figs = 14

    # Each entry: (df_row_idx, x_px, y_px, w_px, h_px) in banner pixel space
    fig_placements = []

    if n_figs == 1:
        # 14 x 4 units – full banner
        fig_placements.append((0, 0, 0, 14 * PATCH_PX, 4 * PATCH_PX))
    elif n_figs == 2:
        # 7 x 4 each
        for i in range(2):
            fig_placements.append(
                (i, i * 7 * PATCH_PX, 0, 7 * PATCH_PX, 4 * PATCH_PX))
    elif n_figs == 3:
        # 4 x 4 each (12/14 total width, 2 units blank)
        for i in range(3):
            fig_placements.append(
                (i, i * 4 * PATCH_PX, 0, 4 * PATCH_PX, 4 * PATCH_PX))
    elif n_figs == 4:
        # 3 x 4 each (12/14 total width, 2 units blank)
        for i in range(4):
            fig_placements.append(
                (i, i * 3 * PATCH_PX, 0, 3 * PATCH_PX, 4 * PATCH_PX))
    elif 5 <= n_figs <= 7:
        # 2 x 4 each
        for i in range(n_figs):
            fig_placements.append(
                (i, i * 2 * PATCH_PX, 0, 2 * PATCH_PX, 4 * PATCH_PX))
    elif 8 <= n_figs <= 14:
        # Two rows of 2-unit height; each figure 2 x 2 units (7 per row)
        for i in range(n_figs):
            row_i = 0 if i < 7 else 1
            col_i = i if i < 7 else (i - 7)
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

    # --- compose 224x224 stacked image ------------------------------------
    stacked = Image.new("RGB", (COMPOSITE_SIZE, COMPOSITE_SIZE))
    stacked.paste(slide_resized, (0, 0))
    stacked.paste(banner, (0, slide_h))

    # --- build 14x14 logical-patch -> original-pixel mapping --------------
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

    slide_frac = GRID_SLIDE_ROWS / GRID_TOTAL_ROWS   # 10/14

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


def predict(slide: Slide) -> list[tuple[int,int]]:
    # get image and OCR data
    image = Image.open(slide.png).convert("RGB")

    # Build 224x224 stacked image (slide + figure banner) and patch mapping
    stacked_image, mapping_info = build_stacked_image_and_mapping(image, slide)

    # Rescale OCR boxes to [0, 999] using original image dimensions,
    # then compress y-coordinates into the top 10/14 of the composite
    # so the model sees OCR positions aligned with the slide portion.
    slide.ocr_text.rescale_bbs_xyxy(
        height=image.height, width=image.width, scale_factor=1000, astype=np.int64
    )
    y_scale = GRID_SLIDE_ROWS / GRID_TOTAL_ROWS   # 10/14
    slide.ocr_text.df['scaled_box'] = slide.ocr_text.df['scaled_box'].apply(
        lambda b: [b[0], int(b[1] * y_scale), b[2], int(b[3] * y_scale)]
    )

    slide.ocr_text.prune_ocr_to_budget(processor.tokenizer, token_budget=200)
    maps = []
    for (sent, tbounds) in slide.asr_text.to_sentences():
        # encode stacked image and sentence
        encoding_slide = processor(stacked_image, sent, return_tensors="pt")
        if encoding_slide['input_ids'][0].shape[0] > 212:
            # Clip the sentence to 208 tokens (leaves room for special tokens) and re-encode
            token_ids = processor.tokenizer._tokenizer.encode(
                sent, add_special_tokens=False
            ).ids[:208]
            sent = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
            encoding_slide = processor(stacked_image, sent, return_tensors="pt")
        encoding_slide |= create_layoutlmv3_input_sequence(slide.ocr_text.df, sent, processor.tokenizer)
        encoding_slide = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in encoding_slide.items()}
        outputs = model(output_attentions=True, return_dict=True, **encoding_slide)

        # Get raw heatmap on the stacked-image grid, project back to
        # original slide coordinates, then normalise.
        raw_heatmap = get_attention_heatmap(outputs, encoding_slide['asr_range'])
        heatmap_np = project_heatmap_to_original(raw_heatmap, mapping_info)
        heatmap_np = heatmap_np / (heatmap_np.sum() + 1e-8)

        maps.append((tbounds, sent, heatmap_np))
        #for w in (set(sent.split()) & set(slide.ocr_text.df['text'].tolist())):
        #    print(f"Match score for {w}: {measure_grounding_score(w, processor.tokenizer, encoding_slide, outputs)}")
    return maps
