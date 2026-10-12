# Project Instructions

Before research or experiment changes, read RESEARCH_CONTEXT.md and configs/assets.json and verify relevant files.

- Preserve the existing FF++ c0 source-aware splits, labels and manipulation provenance.
- Do not regenerate or rewrite frozen manifests for the imported checkpoints.
- Explain the reason and expected impact before changing preprocessing, augmentation, loss, model selection or evaluation protocols.
- Preserve source images and legacy checkpoints/configs; use new versioned outputs.
- Keep original, Deepfakes and Face2Face distinguishable.
- Never use test samples or unseen-filter evaluation data for adaptation or model selection unless a separately documented protocol is explicitly authorized.
- Checkpoint-specific preprocessing is authoritative. Do not substitute DF40 preprocessing.
- Claim only verified completion. Distinguish original/reference results from new runs.
- Keep raw images, weights, secrets and generated outputs out of the public Git repository.
