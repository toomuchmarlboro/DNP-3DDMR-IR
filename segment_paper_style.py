"""
Paper-style segmentation for DMR-IR thermograms.

Implements the pipeline described in:
Trongtirakul et al., 2023 (doi:10.3934/mbe.2023748)

Pipeline (approximation of the published equations):
1) Normalize input image to [0, 1]
2) Local enhancement using a local CDF over a 9x9 neighborhood:
   - Local linear enhancement
   - Local hyperbolization enhancement (beta=0.0001)
3) Image-dependent weighted fusion of the two enhanced images
4) Initial seeds using global threshold (mu=0.8627) + median filter (3x3)
5) Region growing on fused image from seed regions (tolerance epsilon)

Outputs masks as both .npy and .png files, preserving metadata mask paths.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def normalize01(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    vmin = float(image.min())
    vmax = float(image.max())
    if vmax <= vmin:
        return np.zeros_like(image, dtype=np.float32)
    return (image - vmin) / (vmax - vmin)


def _window_view(image: np.ndarray, win: int) -> np.ndarray:
    pad = win // 2
    padded = np.pad(image, pad, mode="reflect")
    return np.lib.stride_tricks.sliding_window_view(padded, (win, win))


def local_cdf(image01: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows = _window_view(image01, win)
    center = image01[..., None, None]

    local_min = windows.min(axis=(-1, -2))
    local_max = windows.max(axis=(-1, -2))

    # Empirical local CDF: percentile rank of center value inside local window.
    cdf = (windows <= center).mean(axis=(-1, -2)).astype(np.float32)
    return local_min.astype(np.float32), local_max.astype(np.float32), cdf


def median_filter_2d(image: np.ndarray, win: int = 3) -> np.ndarray:
    windows = _window_view(image, win)
    med = np.median(windows, axis=(-1, -2))
    return med.astype(image.dtype)


def enhance_and_fuse(image01: np.ndarray, beta: float = 1e-4, win: int = 9) -> np.ndarray:
    local_min, local_max, cdf = local_cdf(image01, win)

    # Eq (2.1): local linear enhancement approximation.
    e_linear = local_min + (local_max - local_min) * cdf

    # Eq (2.2): local hyperbolization enhancement approximation.
    e_hyper = beta * local_max * np.exp(np.log(1.0 + 1.0 / beta) * cdf)

    # Eq (2.3): image-dependent weighted fusion.
    # Using normalized input in [0,1], so I/(L-1) -> I.
    fused = image01 * e_hyper + (1.0 - image01) * e_linear
    return normalize01(fused)


def grow_region_from_seeds(fused: np.ndarray, seeds: np.ndarray, epsilon: float = 0.05) -> np.ndarray:
    h, w = fused.shape
    seed_coords = np.argwhere(seeds > 0)
    if seed_coords.size == 0:
        return np.zeros((h, w), dtype=np.uint8)

    region = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    seed_mean = float(fused[seeds > 0].mean())

    for r, c in seed_coords:
        rr, cc = int(r), int(c)
        region[rr, cc] = True
        q.append((rr, cc))

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    while q:
        r, c = q.popleft()
        for dr, dc in neighbors:
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            if region[nr, nc]:
                continue
            if abs(float(fused[nr, nc]) - seed_mean) <= epsilon:
                region[nr, nc] = True
                q.append((nr, nc))

    return region.astype(np.uint8)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    best_coords = []

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for r in range(h):
        for c in range(w):
            if mask[r, c] == 0 or visited[r, c]:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            comp = [(r, c)]

            while q:
                rr, cc = q.popleft()
                for dr, dc in neighbors:
                    nr = rr + dr
                    nc = cc + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue
                    if visited[nr, nc] or mask[nr, nc] == 0:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
                    comp.append((nr, nc))

            if len(comp) > len(best_coords):
                best_coords = comp

    out = np.zeros_like(mask, dtype=np.uint8)
    for r, c in best_coords:
        out[r, c] = 1
    return out


def read_grayscale(path: Path) -> np.ndarray:
    # Convert all thermograms to single-channel for this classical pipeline.
    image = Image.open(path).convert("L")
    return np.asarray(image)


def save_mask(mask01: np.ndarray, npy_path: Path, png_path: Path) -> None:
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(npy_path, mask01.astype(np.uint8))

    mask_u8 = (mask01.astype(np.uint8) * 255)
    Image.fromarray(mask_u8, mode="L").save(png_path)


def process_one_image(
    image_path: Path,
    out_npy: Path,
    out_png: Path,
    mu: float,
    epsilon: float,
    beta: float,
    local_win: int,
    keep_largest: bool,
) -> tuple[int, int]:
    img = read_grayscale(image_path)
    img01 = normalize01(img)

    fused = enhance_and_fuse(img01, beta=beta, win=local_win)

    seeds = (fused >= mu).astype(np.uint8)
    seeds = median_filter_2d(seeds, win=3)
    seeds = (seeds > 0).astype(np.uint8)

    mask = grow_region_from_seeds(fused, seeds, epsilon=epsilon)

    if keep_largest:
        mask = keep_largest_component(mask)

    save_mask(mask, out_npy, out_png)
    return int(mask.sum()), int(mask.size)


def run_dataset(
    data_root: Path,
    output_root: Path,
    split_filter: str,
    category_filter: str,
    mu: float,
    epsilon: float,
    beta: float,
    local_win: int,
    keep_largest: bool,
    max_images: int,
) -> None:
    splits = ["train", "test", "validation"]
    categories = ["benign", "malignant"]

    if split_filter != "all":
        splits = [split_filter]
    if category_filter != "all":
        categories = [category_filter]

    total = 0
    ok = 0
    failed = 0

    for split in splits:
        meta_path = data_root / split / "metadata.csv"
        if not meta_path.exists():
            continue

        df = pd.read_csv(meta_path)
        if "image_path" not in df.columns or "mask_path" not in df.columns:
            print(f"[WARN] Missing columns in {meta_path}")
            continue

        for _, row in df.iterrows():
            image_rel = str(row["image_path"])
            mask_rel = str(row["mask_path"])

            # Filter by category from the path.
            if not any(f"/{cat}/" in image_rel.replace('\\\\', '/') for cat in categories):
                continue

            total += 1
            image_path = data_root / image_rel
            if not image_path.exists():
                failed += 1
                continue

            if max_images > 0 and ok >= max_images:
                print(f"Reached max images limit: {max_images}")
                print("\nDone.")
                print(f"Total considered: {total}")
                print(f"Succeeded: {ok}")
                print(f"Failed: {failed}")
                print(f"Output root: {output_root}")
                return

            # Save in an output root while preserving original mask relative path.
            out_npy = output_root / mask_rel
            out_png = output_root / Path(mask_rel).with_suffix(".png")

            try:
                mask_pixels, all_pixels = process_one_image(
                    image_path=image_path,
                    out_npy=out_npy,
                    out_png=out_png,
                    mu=mu,
                    epsilon=epsilon,
                    beta=beta,
                    local_win=local_win,
                    keep_largest=keep_largest,
                )
                ok += 1
                if ok % 100 == 0:
                    ratio = 100.0 * (mask_pixels / max(all_pixels, 1))
                    print(f"Processed {ok} images (latest mask area: {ratio:.2f}%)")
            except Exception as ex:
                failed += 1
                print(f"[ERROR] {image_rel}: {ex}")

    print("\nDone.")
    print(f"Total considered: {total}")
    print(f"Succeeded: {ok}")
    print(f"Failed: {failed}")
    print(f"Output root: {output_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper-style thermogram segmentation for DMR-IR")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/local/DMR-IR",
        help="Input DMR-IR root",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/local/DMR-IR-paper-masks",
        help="Output root where generated masks are written",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "test", "validation"],
        help="Dataset split to process",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        choices=["all", "benign", "malignant"],
        help="Category to process",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.8627,
        help="Global threshold for seed initialization (paper default: 0.8627)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Region-growing tolerance on fused intensity",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1e-4,
        help="Hyperbolization constant beta (paper default: 0.0001)",
    )
    parser.add_argument(
        "--local-win",
        type=int,
        default=9,
        help="Local window size for CDF enhancement (paper default: 9)",
    )
    parser.add_argument(
        "--keep-largest",
        action="store_true",
        help="Keep only the largest connected component in the final mask",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum number of images to process (0 = all)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    script_dir = Path(__file__).resolve().parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = script_dir / data_root

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = script_dir / output_root

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    run_dataset(
        data_root=data_root,
        output_root=output_root,
        split_filter=args.split,
        category_filter=args.category,
        mu=args.mu,
        epsilon=args.epsilon,
        beta=args.beta,
        local_win=args.local_win,
        keep_largest=args.keep_largest,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
