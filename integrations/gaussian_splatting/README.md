# Gaussian Splatting Integration

This folder contains the SEGS integration route for a compatible Gaussian text-to-3D codebase.

Files:

- `configs/segs_gaussian.yaml`: paper-aligned prototype config.
- `patches/segs_gaussian_integration.patch`: recovered integration patch containing attention hooks, PCA structural energy guidance, BRISQUE attenuation, and CLIP view guard logic.

Apply patch to a clean compatible upstream checkout:

```bash
git apply /path/to/segs/integrations/gaussian_splatting/patches/segs_gaussian_integration.patch
```

Then run from the SEGS repository root:

```bash
python scripts/run_segs_gaussian.py \
  --integration-dir external/gaussian_text_to_3d \
  --config integrations/gaussian_splatting/configs/segs_gaussian.yaml
```

Attribution for the recovered integration path is listed in `third_party/NOTICE.md`.

