# Celeb-DF V2: 16 frames per video

Source: `Celeb-DF-v2/` (590 Celeb-real, 300 YouTube-real, 5,639 Celeb-synthesis videos).
V1 is not included. Videos are not modified.

Output: `data/celebdf_v2_frames16_v1/`

- `frames/{original-folder}/{video-stem}/{frame-index:06d}.png`: full-frame, native resolution, lossless PNG.
- Each video's `frames.json`: original FPS/frame count, chosen indices, per-frame SHA256, dimensions and source signature.
- `manifest.csv`: all successfully extracted frames; method provenance retained.
- `config.json`: extraction settings, input inventory/test-list hashes and OpenCV version.
- `summary.json`: final counts and decode errors; `complete` must be true before claiming completion.
- `official_test_list.txt`: unchanged copy of the supplied test list.

Sampling: unique rounded linspace across frame index 0 to N-1, up to 16 per video.
No face detection, cropping, resize, JPEG re-encoding, beautification or frame-quality exclusion.
Expected maximum: 104,464 images (14,240 real + 90,224 fake).
Official test pool: 518 videos, expected 8,288 frames. Remaining videos are
marked `non_test_pool`, NOT automatically assigned train/validation roles.
Filename identity fields are not interpreted as source/target directions.
Any future training split requires identity/source-relationship auditing.

The source test list uses 1=real and 0=fake; manifest class_index is 0=real, 1=fake.
Extraction below 3 GiB free disk stops individual work with recorded errors.
Rerunning the same command resumes completed video records and verifies existing outputs
rather than silently overwriting different image content.

```bash
.venv/bin/python -u scripts/extract_celebdf_v2_frames.py \
  --video-root Celeb-DF-v2 \
  --output-root data/celebdf_v2_frames16_v1 \
  --frames-per-video 16 --workers 4
```
