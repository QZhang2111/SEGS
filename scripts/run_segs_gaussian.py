#!/usr/bin/env python
"""Run a Gaussian text-to-3D integration with SEGS guidance."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration-dir", default="external/gaussian_text_to_3d")
    parser.add_argument("--config", default="integrations/gaussian_splatting/configs/segs_gaussian.yaml")
    parser.add_argument("--cuda-visible-devices", default=None)
    args = parser.parse_args()

    train_py = Path(args.integration_dir) / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(f"missing integration train.py: {train_py}")

    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repo_root)
        if not current_pythonpath
        else f"{repo_root}{os.pathsep}{current_pythonpath}"
    )
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    subprocess.run(["python", str(train_py), "--opt", args.config], env=env, check=True)


if __name__ == "__main__":
    main()
