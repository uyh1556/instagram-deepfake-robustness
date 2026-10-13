# Celeb-DF V2 supervised source baselines

## Why a new split

The prior official test includes56 filename celebrity IDs, leaving only3 for completely
identity-disjoint training. The user authorized a custom partition for formal source training.
This experiment **repartitions the complete celebrity pool**, including some images originally
assigned official test. Official-test evaluation is therefore not valid for these checkpoints.
Old data files, splits and FF++ checkpoints remain unchanged.

Split seed42:59 filename celebrity IDs shuffled into47 train,6 validation,6 test (approximately80/10/10).
Ratios concern identities, not frame counts. Validation105 videos, test114 videos.
Only6 test identities limit the breadth of generalization evidence.
Both IDs of a generated video's filename must belong to one split. Cross-split pairs are
excluded, not reassigned. Unknown YouTube identities are excluded. This verifies filename
provenance, not biometric identity. No manipulation direction is inferred.

| split | Real frames | Fake frames | Total |
|---|---:|---:|---:|
| train pool | 7,441 | 54,832 | 62,273 |
| validation | 1,024 | 656 | 1,680 |
| test | 960 | 864 | 1,824 |

Each epoch uses all7441 Real frames and7441 Fake frames sampled without replacement (14882 total).
Fake selection varies deterministically by epoch; both models share sample membership.
Native frozen1.5x crops and original fallback are reused, no smoothing/JPEG/noise augmentation.
Horizontal flip probability0.5 only. Baseline source data are labelled, this is NOT UDA.

## Model conditions

| condition | Xception | CLIP ViT-B/16 |
|---|---|---|
| initialization | ImageNet legacy_xception | OpenAI CLIP pretrained image encoder |
| classifier | new2-class head | Linear512→2 |
| fine-tuning | all parameters | all visual encoder + head |
| input | bicubic299, mean/std0.5 | official bicubic/center crop224, CLIP mean/std |
| AdamW LR | 1e-4 | encoder1e-5/head1e-4 |
| common | 15epochs, batch16, weight decay1e-4, gradient clip1, CUDA AMP, patience4 |

No FF++ detector weights, text encoder, feature L2 normalization or target adaptation.
Pretraining sources and learning rates differ: results cannot isolate CNN-versus-ViT structure.
CLIP version pinned to OpenAI commit a1d071733d7111c9c014f024669f959182114e33.

Best checkpoint: maximum clean validation **video AUC** (mean frame probabilities).
Early stopping requires an improvement greater than0.0001 to reset patience.
Every best AUC update is saved even if below that early-stopping delta.
Test untouched during training, frame/video AUC/F1/accuracy and fixed-threshold0.5 FPR/FNR.
Original class imbalance remains in validation/test; disclose it when discussing accuracy/F1.
Prior Celeb-DF pilot evaluation exposure means this test is not fully untouched final evidence.

## Colab

- notebooks/train_celebdf_xception_seed42.ipynb
- notebooks/train_celebdf_clip_b16_seed42.ipynb
- matching copyable docs/colab/train_celebdf_*_seed42.md

Use existing three whole crop TARs and inventory.json under
MyDrive/instagram-deepfake-robustness/data/celebdf_v2_face_crops16_v1/.
No new data TAR or ZIP upload. Each notebook stores a separate run and evaluates after training.
Shared frozen split in Drive manifests/celebdf_v2_identity_disjoint_v1; models cannot change it.
Training output runs/celebdf_v2_supervised_baseline_v1/{model}_celebdf_identity_v1_seed42.
Last-epoch resume (interrupted epoch repeats), completed-run skip, mismatch abort, low-volume logs.
GPU Colab training and actual pretrained downloads are not claimed locally completed.
Future smoothing/UDA experiments must use the same frozen identity roles and reserve a
training-side unlabelled target subset; do not adapt on these validation/test images.
