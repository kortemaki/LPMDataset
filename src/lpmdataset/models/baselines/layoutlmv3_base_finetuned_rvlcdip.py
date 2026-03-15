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

    # NEW: Normalize so the slice sums to 1 (making it a valid probability distribution)
    # We use sum division rather than softmax because attention is already strictly positive
    heatmap = heatmap / (heatmap.sum() + 1e-8)

    # NEW: Convert to NumPy for downstream metric compatibility
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


processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3Model.from_pretrained("gordonlim/layoutlmv3-base-finetuned-rvlcdip")


def predict(slide: Slide) -> list[tuple[int,int]]:
    # get image and OCR data
    image = Image.open(slide.png).convert("RGB")
    slide.ocr_text.rescale_bbs_xyxy(
        height=image.height, width=image.width, scale_factor=1000, astype=np.int64
    )
    slide.ocr_text.prune_ocr_to_budget(processor.tokenizer, token_budget=200)
    maps = []
    for (sent, tbounds) in slide.asr_text.to_sentences():
        # encode slide
        encoding_slide = processor(image, sent, return_tensors="pt")
        if encoding_slide['input_ids'][0].shape[0] > 212:
            raise ValueError("Cannot process sentence longer than 212 tokens!")
        encoding_slide |= create_layoutlmv3_input_sequence(slide.ocr_text.df, sent, processor.tokenizer)
        outputs = model(output_attentions=True, return_dict=True, **encoding_slide)

        maps.append((tbounds, sent, get_attention_heatmap(outputs, encoding_slide['asr_range'])))
        #for w in (set(sent.split()) & set(slide.ocr_text.df['text'].tolist())):
        #    print(f"Match score for {w}: {measure_grounding_score(w, processor.tokenizer, encoding_slide, outputs)}")
    return maps
