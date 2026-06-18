# Structural Energy Guidance for View-Consistent Text-to-3D Generation

Official prototype implementation for:

**Structural Energy Guidance for View-Consistent Text-to-3D Generation**

Qing Zhang, Jinguang Tong, Jing Zhang, Jie Hong, Xuesong Li

SEGS is a training-free, plug-and-play guidance framework for reducing Janus artifacts in text-to-3D generation. It constructs a structural energy in the PCA subspace of intermediate diffusion U-Net features and injects the energy gradient during denoising, steering samples toward the intended viewpoint without fine-tuning diffusion weights.

This repository currently provides the core SEGS prototype and a Gaussian Splatting integration route. Full benchmark scripts for every baseline are not included in this initial release.

## Highlights

- Training-free guidance: no diffusion fine-tuning or extra 3D training data.
- Structural energy in PCA-projected U-Net features.
- CLIP-based target-view reference selection.
- Optional text consistency guard for viewpoint filtering.
- Gaussian Splatting text-to-3D integration patch.

## Method Overview

SEGS has three main stages:

1. Sample target-view 2D references from a frozen diffusion model.
2. Extract intermediate U-Net structural features and build a PCA subspace.
3. During text-to-3D optimization, minimize structural energy between current rendered-view features and selected target-view references.

Default prototype settings follow the paper implementation:

| Setting | Value |
| --- | --- |
| Reference images | `20` |
| Selected target-view references | `topk=3` |
| PCA feature dimension | `64` |
| Target view | back view |
| Back-view azimuth bin | `azimuth >= 120` |
| Default diffusion backbone | `stabilityai/stable-diffusion-2-1-base` |

## Repository Layout

```text
segs/                               # Core SEGS helper modules
scripts/                            # Public command-line entry points
pca/                                # Legacy reference extraction scripts
integrations/gaussian_splatting/    # Gaussian text-to-3D integration config/patch
examples/                           # Example prompts
third_party/NOTICE.md               # Upstream attribution and dependency notes
```

## Installation

Create an environment compatible with PyTorch, Diffusers, OpenCLIP, and the target text-to-3D backend.

```bash
conda create -n segs python=3.10
conda activate segs
pip install torch torchvision diffusers transformers accelerate open_clip_torch pyiqa pyyaml pillow
```

For Gaussian Splatting integration, install the dependencies required by the target backend and its CUDA extensions.

## Usage

### 1. Build PCA Structural References

```bash
python scripts/build_pca_references.py \
  --prompt "a dragon, back view" \
  --save-folder pca_output/dragon
```

This generates target-view samples and saves `pca_results.pt`.

### 2. Select Target-View References

```bash
python scripts/select_view_references.py \
  --image-dir pca_output/dragon \
  --pca-path pca_output/dragon/pca_results.pt \
  --save-path pca_result/dragon \
  --topk 3
```

This writes `top3_pca.pt`, used by the 3D optimization backend.

### 3. Apply Gaussian Integration

Apply the integration patch to a compatible Gaussian text-to-3D checkout:

```bash
git apply /path/to/segs/integrations/gaussian_splatting/patches/segs_gaussian_integration.patch
```

Run SEGS-guided generation:

```bash
python scripts/run_segs_gaussian.py \
  --integration-dir external/gaussian_text_to_3d \
  --config integrations/gaussian_splatting/configs/segs_gaussian.yaml
```

## Current Release Scope

Included:

- PCA reference extraction.
- CLIP target-view top-k selection.
- Structural energy computation.
- Gaussian Splatting integration patch/config.

Not included yet:

- Full automated benchmark suite.
- Reproduction scripts for every baseline reported in the paper.
- Precomputed PCA reference files or model checkpoints.

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).

Third-party code and model dependencies may have separate licenses. See [third_party/NOTICE.md](third_party/NOTICE.md).

## Acknowledgements

The Gaussian text-to-3D integration route is adapted from EnVision-Research/LucidDreamer. SEGS also builds on the broader SDS/VSD text-to-3D ecosystem, including Diffusers, OpenCLIP, and Gaussian Splatting implementations.

## Citation
The paper can be downloaded from [arXiv](https://arxiv.org/abs/2605.19876).
If you find our paper/code is useful, you could cite:
```
@article{zhang2026structural,
  title={Structural Energy Guidance for View-Consistent Text-to-3D Generation},
  author={Zhang, Qing and Tong, Jinguang and Zhang, Jing and Hong, Jie and Li, Xuesong},
  journal={arXiv preprint arXiv:2605.19876},
  year={2026}
}
```
