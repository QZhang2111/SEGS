#!/usr/bin/env python
"""Select target-view PCA references with CLIP."""

from __future__ import annotations

import argparse
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--pca-path", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    cmd = [
        "python",
        "pca/CLIP_Extract_PCA_Info.py",
        "--image_dir",
        args.image_dir,
        "--pca_path",
        args.pca_path,
        "--save_path",
        args.save_path,
        "--topk",
        str(args.topk),
        "--num_images",
        str(args.num_images),
        "--batch_size",
        str(args.batch_size),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

