#!/usr/bin/env python
"""tit_detector_cli.py

A simple command-line tool that tells you whether a given image contains a tit (the bird) or not.

Usage examples:
    python tit_detector_cli.py --image https://example.com/bird.jpg
    python tit_detector_cli.py --image /path/to/local_photo.jpg
    
    
    python tit_detector_cli.py --image /Users/zarad/Desktop/IMG_0206.jpg
    

The script will download / open the image, run it through the trained CNN from
`train_model.py`, and print a friendly message like:
    Tit Detected (98.7% confidence)

Notes for beginners:
  • It re-uses the *same* preprocessing and model architecture as training.
  • Any http/https URL is downloaded with the `requests` library.
  • Local file paths are supported too.
  • No GUI needed – perfect for quick tests in the terminal.
"""

import argparse
import io
import os
from typing import Union, Tuple

import requests
from PIL import Image
import torch
import torchvision.transforms as transforms

# Reuse architecture & constants from training script
from train_model import SimpleCNN, MODEL_PATH, DEVICE

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_image(source: str) -> Image.Image:
    """Load an image from a URL or local file path and return a PIL.Image."""
    if source.startswith("http://") or source.startswith("https://"):
        # Download image bytes then open with PIL in memory
        resp = requests.get(source, timeout=10)
        resp.raise_for_status()
        img_bytes = io.BytesIO(resp.content)
        return Image.open(img_bytes).convert("RGB")
    else:
        # Treat as local file path
        if not os.path.isfile(source):
            raise FileNotFoundError(f"File not found: {source}")
        return Image.open(source).convert("RGB")


def preprocess(img: Image.Image) -> torch.Tensor:
    """Resize + normalize exactly like during training."""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img)  # type: ignore[arg-type]
    return tensor.unsqueeze(0)  # type: ignore[attr-defined]  # add batch dimension


def predict(tensor: torch.Tensor, model: SimpleCNN) -> Tuple[str, float]:
    """Return (label, confidence 0-1) using the trained model."""
    model.eval()
    with torch.no_grad():
        outputs = model(tensor.to(DEVICE))
        probs = torch.softmax(outputs, dim=1).squeeze()
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()  # type: ignore[index]
    label = "Tit" if pred_idx == 1 else "No Tit"
    return label, confidence


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect whether an image contains a tit bird.")
    parser.add_argument("--image", "-i", required=True, help="Image URL or local file path")
    parser.add_argument("--advanced", "-a", action="store_true", help="Show feature maps for each conv layer")
    parser.add_argument("--range", "-r", nargs=2, type=float, metavar=("VMIN", "VMAX"),
                        help="Fix color scale for feature maps (e.g. -r -3 3). Defaults to -2 2.")
    args = parser.parse_args()

    try:
        pil_img = load_image(args.image)
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        exit(1)

    tensor = preprocess(pil_img)

    # Load trained weights
    if not os.path.isfile(MODEL_PATH):
        print(f"❌ Model weights '{MODEL_PATH}' not found. Please run train_model.py first.")
        exit(1)

    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    label, conf = predict(tensor, model)
    print(f"{label} Detected ({conf*100:.1f}% confidence)")

    # Advanced visualization: plot feature maps
    if args.advanced:
        import matplotlib.pyplot as plt
        import numpy as np

        activations = {}

        # Register hooks for each Conv layer in model.features
        def get_hook(name):
            def hook_fn(_, __, output):
                activations[name] = output.detach().cpu()
            return hook_fn

        # Collect *all* Conv2d layers (even nested ones)
        hooks = []
        layer_count = 0
        for layer in model.features.modules():
            if isinstance(layer, torch.nn.Conv2d):
                hooks.append(layer.register_forward_hook(get_hook(f"conv{layer_count}")))
                layer_count += 1

        # Forward pass to populate activations
        _ = model(tensor.to(DEVICE))

        # Remove hooks
        for h in hooks:
            h.remove()

        # Determine global vmin/vmax once
        vmin, vmax = (-2.0, 2.0) if args.range is None else tuple(args.range)

        # Plot feature maps for each captured conv layer
        for name, fmap in activations.items():
            fmap = fmap.squeeze(0)  # (C,H,W)
            num_ch = fmap.shape[0]
            cols = 6
            rows = int(np.ceil(num_ch / cols))
            plt.figure(figsize=(cols*2, rows*2))
            ims = []
            for i in range(num_ch):
                ax = plt.subplot(rows, cols, i+1)
                im = ax.imshow(fmap[i], cmap='viridis', vmin=vmin, vmax=vmax)
                ax.axis('off')
                ims.append(im)
            # Add a single colorbar (legend) on the right of the grid
            cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
            plt.colorbar(ims[0], cax=cbar_ax)
            plt.suptitle(f"Feature maps of {name}")
            plt.tight_layout()
            plt.show() 