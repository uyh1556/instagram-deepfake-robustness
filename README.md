# Instagram Deepfake Robustness

FF++ raw/c0 기반 Instagram 필터 강건성 연구를 위한 독립 프로젝트입니다.

## 환경

- Python 3.12.8, `.venv`
- `requirements-local.txt`: 기존 프로젝트와 같은 macOS 패키지 98개
- `requirements-colab.txt`: 기존 평가 코드의 Colab 추가 의존성. GPU 런타임의 PyTorch/torchvision을 전제로 합니다.

## 파일 구성

| 경로 | 내용 |
|---|---|
| `data/ffpp_c0_v1/source_images_raw/` | Original 9,181장, Deepfakes 9,158장, Face2Face 9,190장 |
| `manifests/ffpp_c0_source_aware_v1/` | 기존 source-aware split 원본 CSV·그룹 목록·요약 |
| `manifests/ffpp_c0_face_roi_v1/` | 기존 Face ROI의 고정 영역·분할·생성 설정 |
| `checkpoints/ffpp_c0_xception_{standard,letterbox,face_roi}_seed42/` | best.pt, 원본 config.json, 학습 기록 |
| `outputs/reference/` | 기존 원본 테스트 결과·예측 점수 |
| `configs/filters/instagram_app_filters_2026_10_12.txt` | 사용자가 제공한 앱 필터 목록 원문 |
| `configs/assets.json` | 경로·출처·해시·Drive 파일 ID |
| `scripts/` | 기존 baseline 학습·평가·전처리 코드 |

데이터·체크포인트·평가 출력은 Git에서 제외합니다. 기존 프로젝트의 원본은 보존했습니다.

## Colab 데이터

`data/exports/`의 TAR 3개를 Drive의 `instagram-deepfake-robustness/data/ffpp_c0_v1/`에 올립니다.

- `ffpp_c0_original_v1.tar`
- `ffpp_c0_deepfakes_v1.tar`
- `ffpp_c0_face2face_v1.tar`

각 TAR는 `ffpp_c0_v1/source_images_raw/<method>/`로 압축 해제됩니다. `/content`에 풀면 데이터 루트는 `/content/ffpp_c0_v1`입니다.

manifest의 `source_path`와 CSV 바이트를 변경하지 않았으므로, 별도 경로 재작성 없이 데이터 루트와 연결됩니다.

## 원본 평가 재실행 예시

```bash
source .venv/bin/activate
python scripts/evaluate_checkpoint.py \
  --data-root data/ffpp_c0_v1 \
  --manifest manifests/ffpp_c0_source_aware_v1/manifest.csv \
  --checkpoint checkpoints/ffpp_c0_xception_letterbox_seed42/best.pt \
  --output-dir outputs/ffpp_c0_xception_letterbox_seed42/raw_test_recheck_v1 \
  --domain-name ffpp_c0 --batch-size 16 --workers 2
```

Face ROI 모델은 `manifests/ffpp_c0_face_roi_v1/manifest.csv`를 사용합니다. 평가 코드는 체크포인트에 저장된 전처리를 적용합니다.

## 현재 상태

필터 평가·DANN 파일럿: `notebooks/pilot_roi_pilgram2_5filters_da_seed42.ipynb`와 `docs/colab/pilgram2_roi_pilot_v1.md`. 새 Colab에서 저장소를 clone하며 코드 ZIP 업로드는 필요하지 않습니다. 실행 의존성은 `requirements-pilot.txt`입니다.

이미지·manifest·체크포인트·기존 평가 결과를 복사하고 경로·개수·체크포인트 로딩·manifest 해시·TAR 구성원을 확인했습니다. 새 필터 평가나 학습은 아직 수행하지 않았습니다. 대용량 TAR의 Drive 업로드는 사용자에게 넘겼습니다.
