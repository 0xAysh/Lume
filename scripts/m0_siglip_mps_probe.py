#!/usr/bin/env python3
"""M0 probe: load SigLIP 2 towers independently on MPS and embed one image."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, SiglipTextModel, SiglipVisionModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--query", default="girl riding a bicycle")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available; M0 requires Apple Silicon GPU inference.")
    device = torch.device("mps")

    image_processor = AutoImageProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    vision = SiglipVisionModel.from_pretrained(args.model).to(device).eval()
    text = SiglipTextModel.from_pretrained(args.model).to(device).eval()

    image = Image.open(args.image).convert("RGB")
    image_inputs = image_processor(images=image, return_tensors="pt")
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    text_inputs = tokenizer([args.query], padding="max_length", return_tensors="pt")
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    with torch.inference_mode():
        # Warm once so the throughput number excludes first-use MPS compilation.
        vision(**image_inputs)
        text(**text_inputs)
        torch.mps.synchronize()

        start = time.perf_counter()
        image_emb = vision(**image_inputs).pooler_output
        text_emb = text(**text_inputs).pooler_output
        torch.mps.synchronize()
        elapsed = time.perf_counter() - start

    image_emb = torch.nn.functional.normalize(image_emb, dim=-1)
    text_emb = torch.nn.functional.normalize(text_emb, dim=-1)
    image_fp16 = image_emb.to(torch.float16).detach().cpu().numpy().astype(np.float16)
    text_fp16 = text_emb.to(torch.float16).detach().cpu().numpy().astype(np.float16)

    print(f"model={args.model}")
    print(f"device={device}")
    print(f"image_dim={image_fp16.shape[-1]} image_bytes={image_fp16.nbytes}")
    print(f"text_dim={text_fp16.shape[-1]} text_bytes={text_fp16.nbytes}")
    print(f"round_trip_ms={elapsed * 1000:.2f}")

    if image_fp16.shape[-1] != 768 or image_fp16.nbytes != 1536:
        raise SystemExit("unexpected image embedding shape/byte width")
    if text_fp16.shape[-1] != 768 or text_fp16.nbytes != 1536:
        raise SystemExit("unexpected text embedding shape/byte width")


if __name__ == "__main__":
    main()
