#!/usr/bin/env python
"""Build PCA structural references for SEGS."""

from __future__ import annotations

import argparse
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, help="Target prompt, usually with a target-view suffix.")
    parser.add_argument("--save-folder", required=True, help="Output folder for images and pca_results.pt.")
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--num-save-basis", type=int, default=64)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    args = parser.parse_args()

    cmd = [
        "python",
        "pca/PCA_extraction_final.py",
        "--prompt",
        args.prompt,
        "--save_folder",
        args.save_folder,
        "--num_images",
        str(args.num_images),
        "--num_save_basis",
        str(args.num_save_basis),
        "--guidance_scale",
        str(args.guidance_scale),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

