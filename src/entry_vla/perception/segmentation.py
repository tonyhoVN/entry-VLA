"""Box-prompted segmentation with SAM 2.1 (``facebook/sam2.1-hiera-small``).

SAM 3 (``facebook/sam3``, text-promptable) is supported by transformers but its
weights are gated; ``make_segmenter('sam3')`` raises until they are available.
"""

from __future__ import annotations

import numpy as np


class Sam2BoxSegmenter:
    name = "sam2.1"

    def __init__(self, model_id: str = "facebook/sam2.1-hiera-small", device: str = "cuda:0",
                 local_files_only: bool = True):
        from transformers import Sam2Model, Sam2Processor
        self.model_id, self.device = model_id, device
        self.processor = Sam2Processor.from_pretrained(model_id, local_files_only=local_files_only)
        self.model = Sam2Model.from_pretrained(model_id, local_files_only=local_files_only).to(device).eval()

    def segment(self, image: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
        """Return the highest-scoring boolean mask (H, W) for one box prompt."""
        import torch
        from PIL import Image
        inputs = self.processor(images=Image.fromarray(image), input_boxes=[[[float(v) for v in box_xyxy]]],
                                return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model(**inputs, multimask_output=True)
        masks = self.processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0]  # (1, M, H, W)
        scores = out.iou_scores.cpu()[0, 0]
        return masks[0, int(scores.argmax())].numpy().astype(bool)


def make_segmenter(name: str, device: str = "cuda:0"):
    if name in ("sam2", "sam2.1"):
        return Sam2BoxSegmenter(device=device)
    if name == "sam3":
        raise RuntimeError("SAM 3 weights (facebook/sam3) are gated and not downloaded; request access on "
                           "Hugging Face, `hf auth login`, `hf download facebook/sam3`, then add a Sam3 segmenter")
    raise ValueError(f"unknown segmenter {name!r}")
