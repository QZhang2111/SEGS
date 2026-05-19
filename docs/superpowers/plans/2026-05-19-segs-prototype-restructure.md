# SEGS Prototype Restructure Plan

Date: 2026-05-19

## Goal

Turn the recovered code into a public-facing SEGS prototype repository. The repository should expose the paper core:

- structural PCA reference extraction
- CLIP top-k target-view selection
- structural energy guidance
- one Gaussian text-to-3D integration path

Keep attribution for upstream LucidDreamer and related dependencies. Do not present this as a full benchmark reproduction.

## Scope

1. Add a neutral SEGS package with small, testable helpers for view bins, structural energy projection, teacher-feature aggregation, and guidance schedules.
2. Change paper-aligned defaults: top-k 3, 20 reference images, 64 PCA dims, back-view threshold 120 degrees.
3. Add public README, example prompts, integration config, and third-party notice.
4. Keep the existing recovered integration as local evidence, but expose it as a patch/integration route instead of making upstream source the main repository identity.
5. Verify syntax and core helper behavior with lightweight tests.

## Non-Goals

- Full reproduction of all paper baselines.
- Rewriting every recovered script in one pass.
- Deleting recovered ZIP/submodule artifacts before user approval.

## Public Wording

Use "prototype implementation" and "core method." State current release focuses on one Gaussian Splatting integration path.

