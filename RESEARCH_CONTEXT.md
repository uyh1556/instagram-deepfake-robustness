# Research Context

## 현재 연구 방향
FF++ raw/c0로 학습한 Xception에 Instagram 색감 필터를 적용했을 때의 변화와 augmentation/Domain Adaptation을 탐색한다. 현재는 파일 준비 단계이며, seen/unseen 필터 구성·DA 방법·학습 프로토콜은 확정하지 않았다.

## 데이터와 고정 분할
- 기존 프로젝트의 전체 프레임 PNG 27,529장을 복사했다. 원본 영상 전체 또는 FF++ 전체 데이터셋이라고 표현하지 않는다.
- Original → real, Deepfakes·Face2Face → fake. 조작 기법 metadata는 유지한다.
- Custom source-aware split seed42: train 19,267장, val 4,130장, test 4,132장.
- 공식 split이 아니다. 기존 source 연결 관계에 기반한 분할을 그대로 유지했다.
- 기본 manifest SHA256: 3deca3de17f28e9be112e9e6f884fecaaf99fddccaa88355ce3737ef4ea13496
- Face ROI manifest SHA256: 00c5d1e97b245a73a64aa2446595a37153d66d6dfe113cb82df1076a44a5e162
- 경로 root는 `data/ffpp_c0_v1`; CSV의 `source_images_raw/` prefix를 보존했다.

## 재사용 모델
Standard, 전체 프레임 Letterbox299, 고정 얼굴 ROI 1.5배의 seed42 best.pt와 설정·학습 기록을 복사했다. 실제 state_dict 로딩과 checkpoint 내부 manifest 해시를 검증했다. 모델별 기존 crop/resize/normalize를 사용하고, DF40의 canonical256 + Q95 전처리를 자동으로 적용하지 않는다.
원본 config.json의 이전 경로는 출처 기록이므로 수정하지 않는다. 실행 경로는 configs/assets.json과 명시적인 CLI 인자로 설정한다.

## 필터
사용자 제공 61개 항목에는 Normal이 포함된다. configs/filters의 원문을 보존했다. Lo-Fi는 사용자 확인 완료, Willo/Willow 표기는 미확인이다. pilgram2는 Instagram-like 근사 구현으로, 실제 앱과 동일한 처리라고 주장하지 않는다. 필터 적용 데이터는 아직 생성하지 않았다.

## 준비와 검증
configs/assets.json에 출처·경로·Drive ID·이미지 수·manifest 및 checkpoint 해시가 있다. 이미지 경로와 크기, CSV 바이트, ROI와 기본 manifest의 샘플/분할 일치, 모델 로딩 및 TAR 구성원을 확인했다. 모든 이미지의 내용 해시를 전수 비교한 것은 아니다.
Drive에는 모델·manifest·기존 원본 평가 기록을 복사했다. TAR 3개는 사용자 직접 업로드 대상으로 안내했다. 추후 완료 여부는 Drive 파일 목록에서 확인한다.
새 성능 평가나 학습은 수행하지 않았다. outputs/reference는 이전 실험의 결과이며 새 연구 결과로 소개하지 않는다.

## 다음 단계
사용자 TAR 업로드 확인 → 동일 원본 데이터로 baseline 재현 → 필터 파일럿 설계. 학습/평가용 source 분할을 유지하며, augmentation 비교와 DA의 target 데이터 사용 범위를 실험 전에 명확히 한다.

## 필터 파일럿 구현 (2026-10-12)
scripts/pilgram2_roi_pilot.py와 docs/colab/pilgram2_roi_pilot_v1.md, notebooks/pilot_roi_pilgram2_5filters_da_seed42.ipynb를 추가했다. 후보 21개에서 seed42로 Nashville·Lark·Clarendon·Ludwig·Crema를 선택했다. 기본 train/val/test는 기존 split 안에서 기법별 1000/300/500장(총 3000/900/1500장)이다. 전부 0을 지정하면 원본 전체를 사용한다.
전체 프레임에 필터 적용 후 기존 ROI와 resize299를 적용한 lossless PNG cache를 사용한다. 추가 JPEG는 없다. 기존 best.pt에서 시작하는 source_ft·augmentation·DANN을 고정 3 epochs, LR1e-5로 비교하며 원본 validation AUC만으로 best를 선택한다. DANN domain loss 가중치는 0.1, target은 train 이미지의 필터 view이며 해당 view의 real/fake 정답은 사용하지 않는다. 원본 source와 target이 대응하는 모의 적응 설정이다. 5개 필터 모두 적응에 사용하므로 unseen-filter 일반화 실험은 아니다.
tiny CPU smoke만 수행했다. 새 Colab 파일럿의 전체 학습·성능 평가는 완료된 것으로 주장하지 않는다. 설정은 configs/pilgram2_roi_pilot_v1.json, 결과는 새 Drive runs/evaluations의 pilgram2_roi_pilot_v1/ 아래에 저장한다.

## 피부 스무딩 파일럿 구현 (2026-10-13)
기존 paired-view 엔진을 재사용하는 scripts/smoothing_roi_pilot.py와 별도 protocol/Colab을 추가한다.
정확한 segmentation이 아닌 frozen native ROI 내부 기하 마스크(타원+눈/입 제외)를 사용한다.
OpenCV bilateralFilter(sigmaColor35, sigmaSpace ROI 크기의2%)와 원본을 25/50/75% 혼합한다.
얼굴 검출 실패 fallback은 변경하지 않는다. Instagram 공식 알고리즘 재현이 아니다.
원본 split/checkpoint를 유지하며 train3000/val900/test1500, source_ft/augmentation/DANN 각3epochs.
validation은 원본만 사용한다. 새 결과는 smoothing_roi_pilot_v1 아래 저장하고 색감 결과와 분리한다.
모든 강도가 학습에 포함되므로 unseen 보정 일반화 주장을 하지 않는다. 실제 Colab 학습 완료 여부는 결과 파일로 확인한다.

## Celeb-DF V2 프레임/ROI 준비 (2026-10-13)
원본 Celeb-DF-v2 영상6529개에서 full-frame PNG를16장씩 추출하는 작업을 시작했다.
data/celebdf_v2_frames16_v1/summary.json을 확인하기 전 완료로 간주하지 않는다.
공식 test518개는 test로, 나머지는 non_test_pool로 표시하며 새 train/val split은 만들지 않는다.
V1은 추출에 포함하지 않는다. 원본 ZIP/영상과 data 출력을 Git에서 제외했다.
기존 FF++ MTCNN ROI 코드를 create_frozen_face_roi_manifest.py로 재사용하고,
prepare_celebdf_v2_face_roi.py detect/export와 docs/celebdf_v2_face_roi.md를 준비했다.
원본에서 검출한 1.5배 ROI를 보정 전후 동일하게 사용한다. 얼굴 검출 실패는 기존 fallback 유지.
ROI 생성/실제 Celeb-DF crop 실행은 아직 요청하지 않았으므로 자동 실행하지 않는다.

## Celeb-DF 완료 및 UDA 파일럿 준비 확인 (2026-10-13)
사용자 요청으로 V2 전체 frame extraction과1.5x face crop export를 완료했다.
summary와 파일 수를 확인:6529영상,104449PNG. 1프레임짜리 real영상1개 때문에 예상보다15장이 적다.
얼굴 검출101520장, fallback2929장 유지. 크롭PNG 총6825244679bytes.
전체 TAR는 test8288 / non_test_real11377 / non_test_fake84784로 분리해 생성·검증했다.
추가 self-contained pilot TAR에 FF++ source3000/val900과 Celeb target1200/test8288을 넣었다.
target은 공식 test와 filename celebrity IDs가 겹치지 않는 id59/id60/id61만 사용한다.
Target class labels를 adaptation manifest에서 제거했다. YouTube-real identity는 미확인이다.
스크립트celebdf_smoothing_uda_pilot.py에서 baseline/source_ft/source_aug/uda_normal/uda_smoothing 비교.
기존 FF++ ROI best에서 각3epochs, source validation AUC로만 best 선택한다.
native crop에서 smoothing25/50/75% -> 기존299 resize/normalize. Target/test는 재크롭하지 않는다.
local tiny CPU prepare/train/evaluate20조건을 점검했다. 전체GPU 성능 실험은 아직 수행하지 않았다.
pilot TAR1139773440bytes, 전체13388장. 기존 대형 TAR를 추가로 해제할 필요가 없다.
