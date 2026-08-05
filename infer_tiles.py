"""Sliding-window inference for RIFT on high-resolution images."""
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from rift import build_model
from train import get_args_parser

logger = logging.getLogger(__name__)


def build_weight_map(tile: int) -> np.ndarray:
    ramp = np.hanning(tile).astype(np.float32)
    ramp = np.clip(ramp, 1e-3, None)
    return np.outer(ramp, ramp)


def tile_origins(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    origins = list(range(0, length - tile + 1, stride))
    if origins[-1] != length - tile:
        origins.append(length - tile)
    return origins


def apply_clahe(image: np.ndarray, clip_limit: float) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess(tile_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - 0.5) / 0.5
    return rgb.transpose(2, 0, 1)


@torch.no_grad()
def infer_image(
    model: torch.nn.Module,
    image: np.ndarray,
    tile: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    flip_tta: bool,
) -> np.ndarray:
    h, w = image.shape[:2]
    pad_h, pad_w = max(0, tile - h), max(0, tile - w)
    if pad_h or pad_w:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        h, w = image.shape[:2]

    prob = np.zeros((h, w), np.float32)
    weight = np.zeros((h, w), np.float32)
    wmap = build_weight_map(tile)

    coords: List[Tuple[int, int]] = [
        (y, x) for y in tile_origins(h, tile, stride) for x in tile_origins(w, tile, stride)
    ]
    logger.info("tiles=%d (%dx%d image, tile=%d, stride=%d)", len(coords), w, h, tile, stride)

    for start in range(0, len(coords), batch_size):
        chunk = coords[start : start + batch_size]
        batch = np.stack([preprocess(image[y : y + tile, x : x + tile]) for y, x in chunk])
        x_in = torch.from_numpy(batch).to(device)
        logits = model(x_in)
        if flip_tta:
            logits = logits + torch.flip(model(torch.flip(x_in, dims=[3])), dims=[3])
            logits = logits + torch.flip(model(torch.flip(x_in, dims=[2])), dims=[2])
            logits = logits / 3.0
        preds = torch.sigmoid(logits)[:, 0].float().cpu().numpy()
        for (y, x), pred in zip(chunk, preds):
            prob[y : y + tile, x : x + tile] += pred * wmap
            weight[y : y + tile, x : x + tile] += wmap
        if (start // batch_size) % 10 == 0:
            logger.info("processed %d/%d tiles", min(start + batch_size, len(coords)), len(coords))

    prob /= np.maximum(weight, 1e-6)
    return prob[: h - pad_h, : w - pad_w]


def save_outputs(
    out_dir: Path, stem: str, image: np.ndarray, prob: np.ndarray, thresh: float, save_overlay: bool
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    probability_map = prob.astype(np.float32)
    binary_mask = (probability_map >= thresh).astype(np.uint8)

    np.save(out_dir / f"{stem}_prob.npy", probability_map)
    np.save(out_dir / f"{stem}_mask.npy", binary_mask)
    cv2.imwrite(str(out_dir / f"{stem}_prob.png"), (probability_map * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / f"{stem}_mask.png"), binary_mask * 255)
    mask = binary_mask

    if save_overlay:
        overlay = image.copy()
        overlay[mask == 1] = (0, 0, 255)
        blended = cv2.addWeighted(image, 0.55, overlay, 0.45, 0)
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), blended)

        prob_u8 = (prob * 255).astype(np.uint8)
        heat = cv2.applyColorMap(prob_u8, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(out_dir / f"{stem}_heat.png"), heat)
        alpha = (prob[..., None] * 0.85).astype(np.float32)
        heat_blend = image.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha
        cv2.imwrite(str(out_dir / f"{stem}_heat_overlay.png"), heat_blend.astype(np.uint8))

    ratio = float(mask.mean())
    logger.info("crack pixel ratio @%.2f: %.4f%%", thresh, ratio * 100)


def main() -> None:
    parser = argparse.ArgumentParser("RIFT tiled inference", parents=[get_args_parser()])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./results_tiled")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--tile_batch", type=int, default=4)
    parser.add_argument("--flip_tta", action="store_true")
    parser.add_argument("--no_overlay", action="store_true")
    parser.add_argument("--clahe", type=float, default=0.0, help="CLAHE clip limit on L channel; 0 disables.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.load_height = args.tile
    args.load_width = args.tile
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, _ = build_model(args)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    if args.scale != 1.0:
        image = cv2.resize(image, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)

    model_input = apply_clahe(image, args.clahe) if args.clahe > 0 else image
    prob = infer_image(model, model_input, args.tile, args.stride, args.tile_batch, device, args.flip_tta)
    tag = (
        f"s{args.scale:g}_{Path(args.checkpoint).stem}_st{args.stride}"
        + ("_tta" if args.flip_tta else "")
        + (f"_clahe{args.clahe:g}" if args.clahe > 0 else "")
    )
    stem = f"{Path(args.image).stem}_{tag}"
    save_outputs(Path(args.out_dir), stem, image, prob, args.thresh, not args.no_overlay)


if __name__ == "__main__":
    main()
