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


processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ModifiedVilt(ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa"))


def predict(slide: Slide):
    # Get image
    image = Image.open(slide.png).convert("RGB")

    results = []

    # NOTE: You will need to adjust this loop depending on how your
    # slide.asr_text object stores time boundaries.
    # I am assuming a method that yields both the bounds and the text.
    for sent, tbounds in slide.asr_text.to_sentences():

        # Encode slide and sentence
        encoding_slide_asr = processor(
            image,
            sent,
            return_tensors="pt",
            truncation=True,
            max_length=40,
            padding="max_length",
        )
        outputs = model(**encoding_slide_asr)

        # 1. Map out the sequence length
        input_ids = encoding_slide_asr["input_ids"]
        num_text_tokens = 40 #input_ids.shape[1]

        # 2. Map out the image patch grid
        # ViLT dynamically resizes images (usually to 384x384), creating 32x32 patches.
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

        # 6. Collapse into a heatmap and reshape to the grid
        heatmap = text_to_patch_attn.mean(dim=0).view(grid_h, grid_w)

        # 7. Normalize
        heatmap = heatmap / (heatmap.sum() + 1e-8)
        heatmap_np = heatmap.detach().cpu().numpy()

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
