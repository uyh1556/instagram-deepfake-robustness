# Celeb-DF V2 frozen face ROI

**Code preparation only; no Celeb-DF face detections or crops have been run.**
Finish full-frame extraction first. This script requires summary.complete=true.
Start with official test518 videos (expected8288 frames) to evaluate imported FF++ model.
For future training use --scope all with a separate ROI output folder. It does NOT
create an identity-safe train/validation split or claim the non-test pool is leakage-free.

The existing FF++ detector implementation is reused with identical rules:
MTCNN confidence>=0.90, face area2–60%, height/width0.5–2.0,
centerY<=75% of image height, highest-confidence valid detection.
The frozen square ROI is 1.5x the larger bbox dimension; out-of-frame pixels RGB128.
No valid detection -> full-frame letterbox299, retained and counted, no label-dependent dropping.
These inherited rules may reject close-up or low-positioned faces; inspect fallback rates
before deciding on any different protocol. Multi-face choice is confidence-based, not identity-based.

## Step1: coordinates only (recommended first)

From /Users/younghyun/Projects/instagram-deepfake-robustness:

```bash
.venv/bin/python -u scripts/prepare_celebdf_v2_face_roi.py detect \
  --frames-root data/celebdf_v2_frames16_v1 \
  --roi-root manifests/celebdf_v2_test_face_roi16_v1 \
  --scope test --batch-size 8 --device cpu
```

Local .venv already has TensorFlow/MTCNN. On Colab install requirements-roi.txt
and use --device gpu only if TensorFlow can see a GPU.
Detection is resumable. Paths, extraction metadata, labels and test membership are preserved.
Source faces are detected once; smoothing/filter versions must reuse their normalized ROI,
not detect again. A dataset loading full frames with this ROI manifest avoids duplicate crops.

## Step2: optional physical PNG face crops

```bash
.venv/bin/python -u scripts/prepare_celebdf_v2_face_roi.py export \
  --frames-root data/celebdf_v2_frames16_v1 \
  --roi-root manifests/celebdf_v2_test_face_roi16_v1 \
  --crop-root data/celebdf_v2_test_face_crops16_v1
```

Detected crops keep native ROI dimensions, so smoothing can run BEFORE final299 resize.
Original PNGs and videos are not deleted. Additional JPEG encoding is not used.
For imported checkpoint inference: existing bicubic resize299 -> tensor -> Normalize(0.5).
Do not apply frozen ROI again to exported crop_path; crop_manifest keeps original source_path
for provenance and separate crop_path for the pre-cropped files.
The exporter reuses existing PNGs and never silently overwrites them; preserves test metadata.
There is a3GiB low-disk guard. Crops require extra disk, so Step2 is optional.
