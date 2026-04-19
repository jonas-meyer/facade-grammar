"""SAM 3 inference backend.

SAM 3 is phrase-grounded — inputs are image + text (and optionally bounding
boxes), outputs are instance masks + scores + boxes via the processor's
``post_process_instance_segmentation``. Point prompts from SAM 1/2 are not
part of the SAM 3 public API and are omitted here.

``AutoProcessor.from_pretrained("facebook/sam3")`` resolves to the video
processor, which has a different call signature; we load ``Sam3Processor``
directly.
"""

import base64
import io
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor


class SamBackend:
    def __init__(self, model_id: str, *, device: str, dtype: str = "auto") -> None:
        self.device = torch.device(device)
        self.dtype = _select_dtype(dtype, self.device)
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id, dtype=self.dtype).to(self.device).eval()

    def info(self) -> dict[str, str]:
        return {"status": "ready", "device": str(self.device), "dtype": str(self.dtype)}

    @torch.inference_mode()
    def segment(
        self,
        image: Image.Image,
        *,
        prompts: list[tuple[str, list[list[float]] | None]],
        threshold: float = 0.3,
        mask_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Run SAM 3 once per prompt, returning an aligned list of results.

        The ViT vision encoder dominates per-call cost (~6s on RX 6800 bf16)
        and its output depends only on the image, so we run it once on the
        first prompt and reuse the embeddings for the rest — ~3.65x faster
        on a 4-prompt request. transformers' Sam3 doesn't expose a
        first-class "one image, N prompts" batch mode: processor's
        ``text=list[str]`` produces mismatched batch dims vs ``pixel_values``
        and crashes at cross-attention, and ``list[list[str]]`` crashes at
        the tokenizer — so we drive the encoder/decoder split by hand.
        """
        target_size = [image.size[::-1]]
        vision_embeds = None
        results = []
        for text, boxes in prompts:
            proc_kwargs: dict[str, Any] = {
                "images": image,
                "return_tensors": "pt",
                "text": text,
            }
            if boxes is not None:
                proc_kwargs["input_boxes"] = [boxes]
            # BatchFeature.to(dtype=...) casts float tensors only and leaves
            # integer tensors (input_ids, attention_mask) alone. Needed because
            # the vision encoder auto-casts pixel_values to model dtype but
            # the geometry encoder's boxes_direct_project doesn't — a fp32
            # input_boxes + bf16 model weights otherwise crashes.
            inputs = self.processor(**proc_kwargs).to(self.device, dtype=self.dtype)
            pixel_values = inputs.pop("pixel_values")
            if vision_embeds is None:
                vision_embeds = self.model.vision_encoder(pixel_values)
            outputs = self.model(vision_embeds=vision_embeds, **inputs)
            r = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=target_size,
            )[0]
            masks = r["masks"].cpu().numpy().astype(np.uint8)
            results.append(
                {
                    "masks": [_encode_mask(m) for m in masks],
                    "scores": r["scores"].cpu().tolist(),
                    "boxes": r["boxes"].cpu().tolist(),
                }
            )
        return {"results": results}


def _select_dtype(pref: str, device: torch.device) -> torch.dtype:
    if pref != "auto":
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[pref]
    if device.type == "cuda":
        return torch.bfloat16
    # MPS hits an MPSNDArrayMatrixMultiplication "destination != accumulator"
    # assertion on SAM 3 when text + box prompts are mixed, under both fp16
    # and bfloat16. fp32 is the only reliable dtype on Apple Silicon.
    return torch.float32


def _encode_mask(mask: np.ndarray) -> str:
    """Base64-encoded 1-bit PNG — ~10x smaller than RLE for typical facade masks."""
    img = Image.fromarray((mask.squeeze() * 255).astype(np.uint8)).convert("1")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")
