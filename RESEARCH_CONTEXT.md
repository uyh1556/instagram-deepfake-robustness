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

## Celeb-DF supervised baseline 준비 (2026-10-13)
사용자 요청으로 Xception과 CLIP ViT-B/16의 원본 Celeb-DF 정식 학습 코드/코랩을 추가한다.
사용자가 승인한 새 인물 분리 비율80/10/10: filename ID59개를 Train47/Val6/Test6으로 seed42 분리.
기존 공식test 이미지도 새 train에 포함될 수 있는 별도 custom protocol이며 공식test 비교로 보고하지 않는다.
YouTube-real은 ID 미확인이라 제외, fake의 두 ID가 서로 다른 분할이면 해당 영상 제외.
원본 영상/프레임/크롭/이전manifest는 변경하지 않는다. 두 모델은 같은 새manifest를 사용한다.
Train pool Real7441/Fake54832, epoch당각7441, Val1680장/105영상, Test1824장/114영상.
ImageNet Xception299/Normalize0.5와 OpenAI CLIP B16 이미지 인코더224/공식normalize를 사용.
모두 encoder 전체 fine-tuning; JPEG/smoothing/noise/DA 없이 flip0.5만 적용한다.
15epochs, AdamW, batch16, patience4, best는clean validation video AUC로 선택.
기존 FF++ detector를 초기값으로 사용하지 않는다. 새manifest 생성은Colab Cell4에서만 실행한다.
GPU 정식 학습은 아직 실행하지 않았다. Test인물6명 및 기존pilot 평가 노출 한계를 명시한다.

## 원본baseline Drive 결과 및스무딩 평가 준비 (2026-10-13)
Drive의 두 completed/config/history/test_metrics를 확인했다. 같은manifest hash12b45925896c4f088d2864b1093b8d6b2278c1e5a491783932e7859e4d87fe86.
Xception epoch9종료/best5, test frameAUC0.99853094,videoAUC1.0,videoFPR/FNR0.
CLIP epoch8종료/best4, frameAUC0.99530044,videoAUC0.99938272,videoFPR0.18333333/FNR0.
결과는80/10/10 자체test1824장114영상이며 공식test결과가아니다. 로컬best는아직없고Drive저장확인.
요청에따라별도evaluate_celebdf_smoothing.py/Colab을준비한다. 두best고정,0.5threshold고정.
원본및masked bilateral blend25/50/75%,nativecrop에서적용한뒤모델별transform.
fallback그대로,detected-onlyframe추가지표,pairedvideo1000bootstrap및원본재현check.
6명고정conditionalvideoCI이며identitygeneralizationCI가아님. UDA/재학습없음.
실제전체GPU스무딩평가결과는아직없으며Colab에서실행한다. 기존training/smoothing코드는바꾸지않는다.

## 공개탐지기평가준비 (2026-10-13)
사용자요청RECCE+UCF(Xception계열)+SELFI(CLIPB16+ArcFace)를재학습없이추가평가한다.
RECCE/UCF는DeepfakeBench v1.0.1공개가중치, SELFI는저자계정공개609MBstate_dict.
세모델전체checkpoint를실제다운로드하고strictloading·CPU추론점검을완료했다.
RECCE는2-logitDeepfakeBench재현256모델로보정논문의original299/1-logit모델과구분한다.
SELFI는arXiv2506.17592preprint. 공개YAMLnormalizationImageNetmean/std를따른다.
SELFIcheckpoint에학습config없음, README FF++c23/YAML FF-NT차이를protocol에명시한다.
기존MTCNN1.5crop사용으로upstreamalignedcrop과차이있음. 원논문수치재현으로주장하지않는다.
이번외부모델평가는Celeb-DF공식test518영상/8288frames, 기존자체test114영상과구분.
코랩직접!python-u로진행률표시, 원본/25·50·75%4조건, 재학습UDA없음.
전체GPU평가는아직진행하지않았다. setup·prepare·evaluate코랩으로실행예정.

## 공개 탐지기 c23 Aug/UDA custom 파일럿 준비 (2026-10-13)
사용자가 비라벨 target 인물 다양성을 넓히는 자체 분할 파일럿을 승인했다.
기존 create_celebdf_identity_split.partition(seed42)의 80/10/10을 재사용하며 원본 manifest는 변경하지 않는다.
FF++ c23 기존 global_source_aware_v1에서 source train3000(real1500/DF750/F2F750), val600(real300/DF150/F2F150).
Celeb-DF custom train47명에서 정답을 보지 않고3000장(영상당 최대4장), target manifest의 label/class_index는 공란.
Custom test6명1824장 전체, val6명은 사용하지 않음. 공식 test 비교와 구분한다.
TAR data/exports/public_c23_adaptation_custom_pilot_v2/public_detector_c23_custom_source_target_test_seed42_v2.tar는671897600bytes.
SHA256 6e8034c72c777febac7261fbc8c98f8373a92ad84be86210c4227c17ccefd9bf.
실제 TAR의6600 source/val/target +1824 test PNG 경로와 manifest hash를 확인했다.
이전3명 target v1 TAR도 보존했다. 업로드/GPU학습은 아직 실행하지 않았다.
public_c23_adaptation_pilot.py는 source_ft/source_aug/uda_normal/uda_smoothing 각3epochs 및 baseline 평가를 제공한다.
모델별 public preprocessing 유지, BN running statistics 고정, clean source validation frame AUC로 best 선택.
기존 스무딩3강도, domain loss .1 progressive GRL. RECCE/UCF 원논문 보조 손실 재현이 아닌 분류 fine-tuning.
실제 세 공개 checkpoint CPU forward 및 CE+GRL backward와 코랩7셀 구문을 확인했다. GPU 전체 학습 완료 주장은 하지 않는다.
정식 split은 추후 재검토한다. 파일럿의6명 test 및 과거 평가 노출 한계를 명시한다.

## 파일럿 수치 오류 수정 (2026-10-13)
사용자 로그에서 RECCE uda_normal epoch3 batch139/188의 Nonfinite loss 중단 확인. 실제 원인은 로그만으로 미확정.
AMP 비정상 forward loss 발생 시 RNG를 복원해 같은 배치를 FP32 재계산하며 샘플을 건너뛰지 않는다.
FP32도 실패하면 분류/도메인 손실을 별도 기록하고 중단한다. 기존 last는 완료 epoch 기준 보존.
수정 전 코드SHA97858e9e26ef2203c7280acee6e69190eb77513db1c92de20e0bfe17b95b1dcd만 명시적 호환 허용.
다른 데이터/설정/adapter 변경은 거부. config와 checkpoint identity는 보존하고 수치 수정 이력을 별도 저장.
Colab GPU에서 동일 오류가 해결됐는지는 아직 검증하지 않았다.

## SELFI checkpoint 복원 검증 오류 수정 (2026-10-13)
Fine-tuned state mismatch를 실제 공개 SELFI 가중치로 로컬 재현: 제외한 ArcFace925개 key 중 PyTorch는771개만 missing으로 보고한다.
154개 BatchNorm num_batches_tracked를 자동 처리하여 기존 missing_keys 집합 비교가 잘못 실패했다.
저장된 학습 key 집합을 먼저 엄격히 검사하고, 공개 frozen ArcFace state와 합쳐 strict=True로 복원하도록 수정했다.
데이터/학습 손실/모델/전처리 변경 없음. 기존 완료 run과 evaluation을 알려진 코드SHA에 한해 재사용한다.
이전 수치 수정 코드SHA0d4d235b437c3b86e22a2dcf63ece9f9fd480245a8695afca45208e2ffd63a9c도 호환 대상.

## FF++ c23 native ROI16로 교체 (2026-10-13)
사용자 승인: 기존 c23 crop 삭제 및 Celeb-DF와 동일한 MTCNN1.5 native ROI, 영상당16프레임으로 재생성.
scripts/prepare_ffpp_c23_native_roi16.py는 기존 ROI 선택/필터/crop 함수와 rounded linspace를 재사용한다.
전체 프레임 PNG를 저장하지 않고 영상에서 읽어 native crop만 PNG로 저장. 원본 영상과 source-aware split은 보존.
Deepfakes/Face2Face 원본은 /Users/younghyun/Downloads/ffpp_c23 아래 각각1000영상 확인.
Real c23 원본 영상은 확인되지 않았으므로 original 영상 추가 다운로드가 필요하다. 기존 Real crop에서 재생성하지 않는다.
새 출력 data/ffpp_c23_native_roi16_v1. 1영상 smoke16crop 통과 후 전체 두 기법 처리 session8685 시작.
완료 여부는 새 metadata/{method}/summary.json 및 crop_manifest.csv로 확인한다. 예약 자동화 없음.
공간 부족으로 사용자가 재차 명시 승인하여 아래 기존 이미지 폴더를 영구 삭제했다:
/Users/younghyun/Desktop/Projects/deepfake_project/data/ffpp/c23/df40_preprocessed/Deepfakes/ff/frames
/Users/younghyun/Desktop/Projects/deepfake_project/data/ffpp/c23/df40_preprocessed/Face2Face/ff/frames
/Users/younghyun/Desktop/Projects/deepfake_project/data/df40/real/ff/c23/frames
기존 crop 의존 manifest는 provenance로만 보존하며 로컬 경로는 이제 없을 수 있다. 기존 TAR/metadata/landmarks/checkpoints/c0/new crops는 삭제하지 않았다.

## FaceSwap·NeuralTextures ROI16 추가 (2026-10-13)
사용자가 새 두 기법의 프레임추출+ROI를 승인했다. Projects/FFPP_c23에 각1000mp4,0-byte없음,임시파일없음 확인.
Deepfakes complete summary:1000영상/16000crop/fallback1372/PNG2236193955bytes.
Face2Face session8685는 확인 시940/1000 진행. 완료 summary 확인 후 새 두 기법을 순서대로 실행하는 대기 프로세스를 시작했다.
기존과 같은 prepare_ffpp_c23_native_roi16.py, source-aware anchor, Celeb-DF ROI 정책 및 새 공통 출력 폴더 사용.
자동화 예약은 만들지 않음. 실행 중인 정상 처리 프로세스이며 전체 완료 여부는 method summary로 재확인한다.

## Original와 TAR 마무리 및 c0 정리 (2026-10-13)
Original1000mp4/0-byte없음/임시없음 확인. 사용자 승인으로 NeuralTextures 종료 뒤 Original ROI16 및 TAR 생성 파이프라인 session98582 시작.
scripts/finish_ffpp_c23_native_roi16.py -> archive_ffpp_c23_native_roi16.py. 기존sourceID split 유지, 미확인source는 excluded로 분리.
TAR 예정 경로 data/exports/ffpp_c23_4methods_roi16_v1, trainval/test 두 TAR. 공간 검사 실패시 삭제 없이 멈춤.
확인 당시 Deepfakes16000/fallback1372, Face2Face16000/fallback1275, FaceSwap16000/fallback1326 완료; NeuralTextures 약300영상 진행.
사용자가 c0 관련 삭제를 명시 요청했다. 새 프로젝트 data/ffpp_c0_v1와 exports/ffpp_c0_{original,deepfakes,face2face}_v1.tar,
새 프로젝트 checkpoints/ffpp_c0_xception_{standard,face_roi,letterbox}_seed42 및 outputs/reference의 대응 raw_test_v1를 영구 삭제했다.
이전 프로젝트 /Users/younghyun/Desktop/Projects/deepfake_project/source_images_raw c0 사본도 삭제했다.
c0 분할manifest와 설정/코드/문헌 및 Drive는 보존. c23/새ROI/Celeb-DF/공개checkpoint는 보존. 과거 c0 로컬 경로는 더 이상 유효하지 않다.

## FF++ c23 Xception·CLIP·DINOv2 baseline 코드/코랩 (2026-10-13)
사용자 요청으로 모델당1개 코랩, 총3개를 생성했다. scripts/train_ffpp_c23_baseline.py와 requirements-ffpp-c23-training.txt.
노트북/MD stem: train_ffpp_c23_xception_seed42, train_ffpp_c23_clip_b16_seed42, train_ffpp_c23_dinov2_b14_seed42.
새 native ROI16 trainval/test TAR 두 개와 inventory.json을 Drive data/ffpp_c23_4methods_roi16_v1에 사용자 업로드 후 사용.
데이터만 같은 고정 source-aware 분할이며 공식 FF++ split은 아니다. epoch Real:Fake1:1, 네 Fake 기법 균등, 영상 균형 샘플링.
epoch별 목록은 모델에 의존하지 않고 seed42+epoch*100003으로 고정; 목록hash를 history에 저장한다.
첫1epoch 분류기만 학습, 이후 Xception 전체 / CLIP 마지막2블록+ln_post+visual projection / DINO 마지막2블록+final LayerNorm을 학습.
Xception encoder/head LR1e-4; CLIP/DINO encoder1e-5, head1e-4. 최대15epochs, patience4, AdamW, batch16.
기준 입력: Xception299 Normalize0.5; CLIP224 OpenAI 공식transform; DINO224 direct bicubic + ImageNetmean/std.
DINO는 facebook/dinov2-base revision f9e44c814b77203eaa57a6bdbbd535f21ede1415, CLS768 -> Linear2, patch14.
CLIP은 OpenAI visual512 -> Linear2; text encoder/추가featureL2 normalize 없음. 사전학습에서 시작하며 랜덤 대체 없음.
학습 flip0.5만, validation/test 변형 없음. 최고checkpoint는 FF++ clean validation macro4 frame AUC로 선택.
BF16지원GPU는BF16, 그 외FP32; FP16사용안함. finite loss/gradient 검사, strict checkpoint load, last epoch재개/완료skip.
각 코랩에 FF++ test 평가 포함. AUC/FPR/FNR/F1/Accuracy 및 기법별 지표 저장, video지표 없음.
Celeb-DF/스무딩/augmentation/UDA는 포함하지 않으며 후속 평가/학습 단계로 분리한다.
tiny합성fixture에서 epoch중단/재개/완료skip/test재사용 검증, 세 백본 architecture·transform·warmup·부분gradient 및18개코드셀 구문 확인.
실제 공식 DINOv2 가중치 로딩 및224입력 forward/backward도 확인. FF++ GPU정식학습은 아직 실행하지 않았다.

## FF++ 세 baseline의 고정 Celeb-DF 공식 스무딩 평가 (2026-10-14)
사용자 승인: Celeb-DF 공식 test를 공통 평가로 고정, 같은 인물은 허용하되 같은 영상/프레임은 적응에 사용하지 않는다.
evaluate_ffpp_baselines_celebdf_smoothing.py 및 동명 평가 Colab/MD 추가. 완료된 FF++ c23 Xception/CLIP/DINO best.pt만 사용.
공식 test TAR 557015040bytes,518영상/8288장(Real2848/Fake5440) 확인. 원본+기존25/50/75% bilateral 기하마스크 스무딩.
모델별 기존 전처리 유지, 추가JPEG 없음, fallback변형안함, FP32 추론, frame지표만, threshold0.5.
평가 manifest/프로토콜/영상제외목록과 checkpoint해시/조건별 예측을 저장하여 향후 동일조건 baseline 재평가를 줄인다.
원본–조작 영상 연결 중복 검사는 향후 UDA 준비 시 별도 필요하며 완료했다고 주장하지 않는다.
실제 Colab baseline 학습 완료 및 새로운 smoothing 성능 결과는 아직 원격 파일로 확인하지 않았다.
로컬 tiny fixture에서 스무딩 생성/추론/완료결과 재사용/fallback 유지/평가영상 목록 저장 통과. 코랩6개 코드셀 구문 검사 통과.

## FF++ baseline 추가학습 Aug/DANN 비교 (2026-10-14)
사용자 요청: baseline 평가와 별도 코랩에서 추가학습을 먼저 실행. 최대10epochs/patience3 선택, 검증1200장 승인.
ffpp_c23_smoothing_adaptation.py, train_ffpp_c23_smoothing_aug_uda_seed42 MD/노트북 추가.
source_ft/augmentation/uda_normal/uda_smoothing을 동일 초기 FF++ baseline에서 비교. 기존 source split/model/preprocessing 보존.
source epoch6000장(Real3000/Fake각750), source clean validation1200(Real600/Fake각150) 고정부분집합.
UDA target5000장 정답 제거 후 영상균등 선택/반복샘플링. 공식 test 제외 및 양쪽identity_same_suffix 원본후보 연결component 보수적제외.
로컬 전체목록에서 conservative eligible342영상/5472장. filename후보이며 실제source/driver관계를 완전검증했다고 주장하지 않는다.
분류CE+0.1DANN(progressiveGRL), target정답/pseudo-label없음. LayerNorm domainhead, 모든전략BN통계고정, BF16지원시BF16/아니면FP32, FP16없음.
Xception전체/CLIP·DINO기존마지막2블록 등 재사용, LR1e-5/1e-5. checkpoint는 추가학습epoch중 source clean val macro4AUC.
평가도 포함: 공식test 원본/스무딩3단계 FP32 frame지표만. baseline 저장결과 재사용, baseline 재추론없음.
prepare_ffpp_c23_training_data.py에 --trainval-only 추가(기존기본동작은그대로), 이코랩은FF++testTAR제외.
로컬 tinyfixture 네전략학습/완료skip/target라벨제거/고정source목록/평가재사용 검사통과. 실제3백본 DANN forward/logit동일성/유한gradient 검사통과.
실제Colab 추가학습은 실행하지 않았다. 새TAR생성/Drive변경/자동예약/Gitpush없음.

## Supervised DA 전용 비교 준비 (2026-10-14)
사용자 요청: 세 FF++ baseline에 supervised DA를 적용하고 SDA 결과만 평가, 불필요한 기존학습/평가 제외.
scripts/ffpp_c23_supervised_da.py, train_test_ffpp_c23_supervised_da_seed42 MD/노트북 추가.
sda_normal/sda_smoothing만 제공: FF++ source CE + 같은 UDA target5000장의 실제정답 CE, 각weight1. DANN/클래스가중치없음.
기존UDA manifest/protocol/source_validation을읽고 재생성목록과동일성 확인, 비테스트metadata에서정답복원. 비율/순서/영상샘플러 보존.
source6000/epoch, FF++ val1200, 최대10/patience3, LR1e-5, 기존backbone trainable범위·전처리·BN고정 유지.
공식test8288 원본/스무딩3단계에서SDA6개체크포인트만평가. Baseline/FT/Aug/UDA 및FF++test는재학습/재평가안함.
새TAR/Drive변경/Gitpush/자동화없음. 작은fixture에서 라벨복원/기존UDA와같은source·targetview/학습/완료skip/중단재개/평가재사용 통과.
코랩7개코드셀 구문검사 통과. 실제SDA GPU학습은 아직 실행하지 않았다.

## 이전 추가학습 결과 확인 메모 (2026-10-14)
앞선Drive읽기에서 세모델5전략4조건=60개평가행확인. DINO augmentation은원본/strongAUC0.7739/0.7671,
baseline0.7281/0.7173 및source_ft0.7459/0.7405보다높음. CLIP UDA_smoothing 원본FNR89.71%,strong80.94%로실제쪽판정편향.
target_unlabelled.csv 5000장/342영상,정답컬럼없음. 사후출처집계 Real3914/Fake1086(78.28%/21.72%),YouTube-real3357장.
이 불균형은원인후보이나인과관계확정아님. 기존문단의GPU미실행은코드작성당시기록이며,해당baseline/추가학습후평가결과는이후Drive에서확인했다.

## 저장된 예측 기반 로짓 이동 분석 준비 (2026-10-14)
사용자 요청: analyze_smoothing_logit_shift.py 및 analyze_baseline_smoothing_logit_shift_seed42 MD/코랩 추가.
기본은세baseline의원본/스무딩CSV·JSON12쌍만읽음. 학습/추론/이미지/TAR/checkpoint/GPU불필요.
crop_path별pair로 실제/가짜·전체/검출성공/fallback별 Δ확률, 확률상승비율, Δmargin중앙값/평균/IQR, 판정기준교차율 계산.
margin=log(p)-log1p(-p), 저장softmax확률기반환산이며rawlogit직접저장값이아님. 정확한0/1pair는margin통계에서제외, 포화율표시, clipping없음.
검출성공이미지에대한marginhistogram과원본/strong확률분포PNG저장. frame기술통계이며영상상관을무시한유의성주장없음.
원본평가파일변경없음. 작은fixture에서순서다른CSVpair연결/포화제외/라벨별통계/PNG출력/재실행/코랩4셀구문검사통과.
실제전체prediction 분석은아직실행하지않았다. Gitpush/Drive변경/자동화없음.

## SDA 결과 및 클래스 보정 비교 준비 (2026-10-14)
Drive에서SDA 3모델×2전략×4조건=24개평가JSON을확인했다. test8288장/manifest해시동일/threshold0.5/FP32.
세모델모두baseline보다SDA AUC가높고FPR은낮지만FNR은높다. target Real3914/Fake1086 불균형은원인후보이며인과확정아님.
사용자승인으로scripts/ffpp_c23_balanced_supervised_da.py 및train_test_ffpp_c23_balanced_sda_seed42 MD/코랩추가.
기존SDA 스크립트변경없이입력복원/스무딩view/영상샘플러/전처리/체크포인트읽기/평가엔진을재사용한다.
새조건sda_smoothing_balanced만세모델각1개. 기존FF++baseline부터시작하며SDA완료모델을계속학습하지않는다.
같은frozen target5000장의영상균등샘플러classprior p로weight=1/(2p) 계산. target CE만per-image weight 후배치mean,
배치weight합으로나누지않아단일class배치에서도보정유지. source CE그대로, 기대target평균weight1.
source6000/epoch, val1200, 최대10/patience3, seed42, 기존trainable범위/BN고정/검증macro4AUC유지.
새runs/manifests/evaluations:ffpp_c23_balanced_supervised_da_v1. 공식test원본/스무딩3단계12조건만새추론.
비교셀은기존SDA smoothing 저장예측을읽어AUC/FPR/FNR/ROC상TPR@FPR<=5%와차이저장,기존결과재추론없음.
ROC지표는기술통계이며test기반배포threshold선택없음. 대상영상source/driver관계는기존filename후보제외한계유지.
tinyCPU에서video-prior/fixed-loss스케일/동일view/train-skip-resume/eval-reuse/순서다른paired비교/코랩7셀구문통과.
실제클래스보정GPU학습은아직실행하지않았다. 새TAR/Drive변경/Gitpush/자동화없음.

## Aug·혼합 타깃 UDA·Aug+UDA 균형 비교 준비 (2026-10-14)
사용자 승인으로 scripts/ffpp_c23_mixed_balanced_uda.py 및 train_test_ffpp_c23_mixed_balanced_uda_seed42 MD/코랩 추가.
세 FF++ baseline에서 augmentation/uda_mixed_balanced/uda_aug_mixed_balanced 각1개, 총9개 run을 새로 학습한다.
기존 UDA 고정 target5000장 및 FF++ val1200장을 재사용한다. target 비테스트 정답은 배치 Real/Fake5:5 샘플링에만 사용하고 loss/모델에는 전달하지 않는다.
클래스 내부 영상균등 반복선택이며 source6000장당 target6000회(Real3000/Fake3000), 고유이미지5000장 자체를5:5로 바꾸는 것은 아니다.
타깃은 원본/약/중/강 각25% 무작위 선택, 1회노출당1view, 보정정보/원본짝 미제공. fallback unchanged.
Aug 및Aug+UDA source는 clean+random3strength; UDA source는clean. source epoch순서와 UDA 두전략의 target경로/변형계획 동일.
Aug+UDA의 source2view/target1view domain loss는각domain평균후1/2씩 합산. 추가view로계산량더큼, 동일계산량주장없음.
최대10/patience3/seed42/batch16/LR1e-5/BN고정 및clean source validation macro4 AUC선택 유지.
기존공식test8288장×4조건으로새9모델36조건평가, baseline저장결과재사용. FPR/FNR/평균오류율/AUC frame지표.
기존source split/metadata/코드/결과는보존하며새 ffpp_c23_mixed_balanced_uda_v1 출력사용. 새TAR/Drive변경/Gitpush없음.
작은CPU fixture에서배치정확한균형/4조건분포/동일노출/손실계산/세전략학습·완료skip/중단후정확한가중치재개/평가재사용/코랩7셀구문 통과.
전체3백본GPU학습은실행하지않았다. 기존UDA대비변형혼합과샘플링을함께바꾸므로각변경의개별인과효과는분리하지못한다.

사용자 후속 요청으로 새 학습에서Aug를 제외했다. 기본실행은세모델×UDA/Aug+UDA=6run,공식test새추론24조건.
코랩마지막셀에서기존 ffpp_c23_smoothing_adaptation_v1/seed42 하위Aug JSON을동일평가프로토콜확인후재사용하며비교CSV저장.
기존Aug와신규학습배치순서차이는참고비교의한계로명시한다.

## B: 고정 탐지기의 스무딩 판단 보존 실험 준비 (2026-10-15)
사용자 승인으로 scripts/ffpp_c23_smoothing_preservation.py 및 train_test_ffpp_c23_preservation_b_seed42 MD/코랩 추가.
세 FF++ c23 baseline의 인코더·원래 분류기·정규화 통계를 전부 고정하고, 32차원 hidden의 작은 residual margin 모듈만 학습한다.
마지막 층0초기화. scale=max(1, 첫 source epoch clean |margin| q75), 보정 한도±2scale, 특징LN(affine없음)+margin/scale을 사용.
손실은 clean/smoothed 평균BCE + normalized Huber(smooth corrected, frozen clean) + normalized Huber(clean corrected, frozen clean), 가중치각1.
기존 source6,000/epoch와 seed별 회전/영상균형/paired flip, 약중강 무작위 스무딩을 유지. 기존 동일 FF++ val1,200장×4조건.
선택은4조건평균BER최소로 변경하며 epoch0도 후보. cleanBER+1%p, cleanFPR/FNR각+2%p, clean/평균macroAUC−0.005 이내만 허용.
기존 Aug의 cleanAUC 선택과 다르므로 참고 비교이며, B항목 기여 확인에는 동일 선택의 A가 추후 필요.
최대10epoch/patience3/seed42, batch16, 보정모듈AdamW LR1e-3. frozen FP32 특징을 로컬에 캐시하고 원본 데이터/기존run은 변경하지 않는다.
CDF학습 없음, non-test TAR불필요. 기존 공식test TAR1개로8,288장×4조건평가, baseline/Aug저장결과 재사용.
새runs/manifests/evaluations는ffpp_c23_smoothing_preservation_b_v1. 작은모듈만checkpoint에저장해원래baseline도필요하다.
로컬tinyCPU로3모델경로 학습/완료skip/epoch0선택/본체고정/손실계산/동일가중치중단재개/평가재사용/fallback/코랩8셀 검사통과.
실제3백본구조에서도 특징2048/512/768,0보정출력동일,보정모듈역전파와본체gradient없음을확인했다(랜덤가중치구조검사).
실제 GPU정식B학습은 아직 실행하지 않았다. Gitpush/새TAR/Drive수정/자동화없음.

## CLIP B epoch3 진단 평가 추가 (2026-10-15)
Drive 완료/history 확인: CLIP B는3epoch종료/선택epoch0. Source4조건BER는21.48→17.31/17.42/16.13%였으나,
cleanFNR7.67→13.50/12.67/14.67%로각각+2%p제한을넘어후보가모두제외됐다. CDF에선epoch0을평가하여Baseline과동일.
사용자가epoch3평가를요청하여evaluate_preservation_b_epoch.py와같은코랩Cell9를추가했다.
기존학습스크립트SHA25e392891daabde15bc3bbfb046de4a4c0119cbe2886efd074889aafb3811228을그대로보존한다.
last.pt실제epoch를검사한뒤작은모듈만로컬임시폴더에서기존평가엔진으로읽어평가하며원본best/last/completed는변경하지않는다.
새출력diagnostic_epoch3에고정epoch진단임을기록한다. 선택기준변경/재학습/테스트기반best선택없음.
tinyCPU에서고정epoch평가·재사용·기존파일해시보존·잘못된epoch거부·9코랩셀검사를수행한다. 실제GPU epoch3평가는아직없음.

## C: 학습된 B에서 비라벨 타깃 일관성 추가 (2026-10-15)
Drive후속결과확인: B Xception5epoch종료/best2, DINO9epoch종료/best6, CLIP3epoch종료/best0.
사용자가 C 진행과 'B best epoch0이면 last.pt에서 시작, 그 외 best.pt' 규칙을 명시 승인했다.
ffpp_c23_smoothing_consistency_c.py 및 train_test_ffpp_c23_consistency_c_seed42 MD/코랩7셀 추가.
따라서현재시작점Xception best2/CLIP last3/DINO best6. B파일/기존학습코드SHA는보존한다.
본체·원래분류기고정/보정모듈만학습. B source손실+0.1normalizedHuber(target두view)+1normalizedHuber(observedtarget점수,frozenB시작점점수).
CDF기존frozen5,000장을재사용, 기존GT보조5:5샘플러6,000회/epoch. 라벨은샘플러에만쓰며모델/손실에전달하지않는다.
Observed는원본/약/중/강각25%; second는observed에strength.25를추가. 이미강한스무딩인입력에도약한추가변형을적용한다.
깨끗한target원본짝을제공하는설정아님. fallback유지. 일관성만으로올바른절대점수/경계를식별할수없다는한계를명시한다.
source6,000+target6,000회, 각2view로24,000views/epoch. 추가계산량은B보다크다. 새AdamW LR1e-4, 최대10/patience3/seed42.
기존동일sourceval1,200×4조건BER와guard를C시작B대비적용. Cepoch0은학습된B이며보정값0 baseline이아니다.
평가는같은CDF공식test8,288×4조건에서B실제시작모델과C를한번의본체forward로비교. Baseline/Aug저장결과재사용.
C-vs-B차이에는추가source학습효과도포함되며타깃항목의독립기여로단정하지않는다. 새TAR없음/Gitpush없음/Drive변경없음.
tinyCPU로best/last분기,3모델학습/완료skip,정확한재개,5:5샘플링,스무딩중첩,손실계산,teacher고정,평가재사용,B파일보존,7셀구문검사통과.
전체GPU C학습은아직실행하지않았다.

## C 독립 초기화로 변경 — 사용자 요청 (2026-10-15)
위 순차 C v1 설계는 새 실험에서 사용하지 않는다. C는 B가 아니라 원래 FF++ Baseline에서 독립적으로 시작한다.
프로토콜/새 출력 경로: ffpp_c23_smoothing_consistency_c_independent_v2. 기존 B와 v1 결과는 보존한다.
새 zero-output 보정 모듈, scale/bound 초기화, LR1e-3, source 이미지·배치 순서, 최대10epoch/patience3과 FF++ 검증 규칙을 B와 맞춘다.
손실은 B의 source 손실 +0.1 target pair normalized Huber. B teacher와 target 점수 유지 항목은 제거했다.
기존 CDF 5,000장과 GT 보조 5:5 노출 샘플링/observed+추가약한스무딩 두 view는 유지한다. 타깃 정답은 분류 손실에 쓰지 않는다.
검증 guard는 원래 Baseline 대비 적용. 선택 epoch0은 Baseline이며 이 경우 last.pt를 별도 진단으로 자동 평가하고 best를 대체하지 않는다.
B 비교는 저장된 검증 선택 평가만 읽는다. CLIP B epoch3 진단을 정식 best로 대체하지 않는다.
source 최대 예산은 같지만 실제 조기종료 시점은 다를 수 있고 target feature 추출 추가 계산량은 존재한다.
CPU tiny-model 검사: 3모델 학습/완료 skip/정확한 중단 재개, source 배치 순서, target 5:5 및 손실, 평가 재사용, Baseline 보존, 코랩7셀 구문 검사를 통과했다.
전체 GPU 독립 C 실험은 실행하지 않았다. Git push 및 Drive 변경 없음.

## C 세 모델 BER-only 선택 실험 (2026-10-15)
사용자가 CLIP을 포함한 전체 세 모델의 선택 제한 완화를 승인했다.
C에 --selection-policy ber_only를 추가했다. 기존 기본값 guarded는 유지한다. BER-only에서는 원본 BER/FPR/FNR 및 원본/평균 AUC guard를 모두 비활성화한다.
FF++ 검증 1,200장×4조건 평균 BER만으로 best와 조기종료를 결정한다. epoch0 후보, patience3/min_delta0.0001, 최대10epoch는 유지한다.
새 MD/코랩 train_test_ffpp_c23_consistency_c_ber_only_seed42는 세 모델 모두 Baseline부터 다시 학습하고 평가한다.
출력 roots는 ffpp_c23_smoothing_consistency_c_ber_only_v3로 분리하며 run basename은 내부 independent_v2 프로토콜 이름을 유지한다. config.selection에 실제 policy를 기록한다.
기존 B/C 데이터·checkpoint는 변경하지 않았다. 기존 B는 guard가 있으므로 새 C와 동일 선택 기준의 공정 비교로 해석하지 않는다.
코드 SHA가 바뀌므로 과거 C 학습 재개/평가에는 당시 코드가 필요하다. 새 코랩은 새 경로에서 시작한다.
변경은 관찰 결과를 보고 정한 탐색적 규칙임을 문서화했다. GPU 학습/Drive 변경/Git push는 수행하지 않았다.

## 기존 C 정리 및 코랩 이름 통합 (2026-10-15)
사용자의 명시적 삭제 요청으로 Drive 프로젝트의 runs/manifests/evaluations 아래 ffpp_c23_smoothing_consistency_c_independent_v2 폴더 3개를 영구 삭제했다.
삭제 폴더 ID: runs=1jFiVTEG4qpvl0gYJ6y2ZzCYKOpDIvgE3, manifests=15yy4a2ghQGu5K9wf7p6D2d1z3hBAkS3J, evaluations=135ZCMtl-r_l2rMbIg49lSoUMpjew3X-n.
Baseline/B/공유 데이터와 신규 ber_only_v3 출력 경로는 삭제하지 않았다. Colab 런타임의 실행 프로세스·로컬 캐시는 원격으로 정리하지 않았다.
BER-only MD/코랩 내용을 기존 train_test_ffpp_c23_consistency_c_seed42 이름으로 통합하고 별도 _ber_only_seed42 파일은 제거했다.
현재 기존 C 이름의 코랩이 세 모델 전체 --selection-policy ber_only를 사용하며 출력은 ber_only_v3로 분리한다. Git push는 하지 않았다.

## 스무딩 반응 분류 학습 — CLIP 먼저 실행 (2026-10-16)
사용자 요청으로 ffpp_c23_smoothing_response.py와 train_test_ffpp_c23_clip_smoothing_response_seed42 MD/코랩을 추가했다.
CLIP은 실행 순서상 먼저이며, 스크립트는 Xception/DINOv2에도 같은 설계를 지원한다. 새 실험 명칭에는 pilot을 사용하지 않는다.
Current-only와 Response 모두 FF++ baseline encoder·기존 classifier를 완전히 고정하고 hidden32 GELU MLP만 학습한다.
MLP 입력은 LN(current features), feature delta/global training RMS, current margin/scale, margin delta/scale이며 scale은 첫 학습 표본에서만 계산한다.
현재 margin+MLP residual로 판정한다. 마지막 층0초기화, 기존 clean margin 복원/teacher/consistency 손실 없이 실제 source 정답 BCE만 사용한다.
Current-only는 delta 입력을0으로 가려 같은 구조·초기화·parameter 수·학습 목록·순서로 비교한다. 유효 입력 자유도는 다를 수 있다.
기존 source6,000/epoch·validation1,200과 source-aware 분할을 재사용한다. clean+random25/50/75% observed 두 view 각각에 약한25% probe를 추가한다.
Native ROI·기존모델별 전처리·fallback 유지. 대응 flip 동일, FP32 eval-mode 본체, source rotation seed42+epoch*100003.
특징 추출은 두 조건이 공유하며 epoch당12,000 observed+12,000 probe forward. 배포 Current-only1회/Response2회 forward이므로 동일 계산량 주장은 하지 않는다.
CDF training/target 비테스트 TAR 불필요. CDF 공식test8,288장×원본/약/중/강 평가에서 입력이미지와 그 입력의 추가 probe만 사용한다.
최대10epoch/patience3/AdamW LR1e-3/batch16, 4조건 source validation 평균BER만으로 best선택, epoch0후보/추가guard없음.
epoch0선택시 last는 별도 진단으로 표시하고 best는 유지. 분류기와 local feature cache 재사용 및 완료epoch 재개 지원.
새 출력 roots는 ffpp_c23_smoothing_response_v1. Baseline은 필요한 frozen forward에서 함께 측정하고 기존 Aug는 호환되는 저장 결과만 참고한다.
CPU 작은 fixture에서 실제 train/evaluate·재개 가중치 일치·완료skip·평가재사용·반응mask·fallback을 확인했고, 코랩7개 코드셀 구문을 확인했다.
실제 CLIP GPU 학습 및 성능 평가는 아직 실행하지 않았다. 기존 데이터·모델·결과 변경, Git push, Drive 변경은 수행하지 않았다.

## Response 평가 후 기존 Aug 집계 오류 수정 (2026-10-16)
사용자 로그에서 CLIP Baseline/Current-only/Response의4조건 평가 완료 후 Aug 비교 단계 KeyError('images')를 확인했다.
기존 ffpp_c23_smoothing_adaptation.py는 Aug JSON에 flat 지표·identity·checkpoint·unit·threshold만 저장하며 images/precision은 저장하지 않는다.
학습/평가 스크립트의 SHA 0c963c634fc850d99bf8cc388683f44004b32ec0699d2d6b4b34d5a4f1aebe85를 그대로 유지하여 기존 체크포인트·특징캐시·평가 identity가 바뀌지 않게 했다.
summarize_smoothing_response.py를 추가하여 저장된 JSON/CSV만 읽고 비교표를 완성한다. 과거 Aug의 누락 필드는 원본 파일에 추가하지 않으며,
동일 evaluation protocol·test manifest·prediction CSV의 영상/이미지/라벨/행 수를 확인한다. Aug가 호환되지 않아도 primary 결과는 요약한다.
코랩 Cell6은 evaluate에서 --reference-evaluations를 제외하고 별도 집계 명령을 이어 실행하도록 수정했다. 이미4조건 평가를 마친 실행은 집계 명령만 실행한다.
CPU regression에서 images/precision 없는 실제 과거 형식의 Aug 병합·추론 호출 없음·기존 예측 보존을 확인했고,7개 코랩 코드셀 구문을 확인했다.
사용자 Colab 런타임/Drive 결과를 직접 수정하거나 GPU 재평가하지 않았다. Git push는 하지 않았다.

## Source-only 클래스별 스무딩 점수 의존 억제 구현 (2026-10-16)
사용자 승인으로 ffpp_c23_smoothing_pivot.py 및 train_test_ffpp_c23_smoothing_pivot_seed42 MD/코랩을 추가했다.
CLIP부터 실행하며 MODELS로Xception/DINOv2도지원한다. 기존 FF++ baseline부터각각독립시작하는 새 Aug와 conditional_pivot을 비교한다.
FF++ source6000/epoch, Real3000/Fake각750, 동일 seed별영상균형목록·배치·원본+랜덤약중강view·대응flip을 유지한다.
Xception전체/CLIP·DINO기존마지막2블록 등과분류기학습, BN통계고정, encoder/headLR각1e-5.
후보만 training-only 진짜/가짜별2개 Linear1→32GELU→4 보조판별기로 margin/10의스무딩강도를예측한다.
탐지기에는gradient reversal을적용해진위CE를유지하면서클래스내강도의존을줄인다. reversal최대0.1/2epoch선형증가, criticLR1e-3.
원본/약/중/강노출prior1/2·1/6·1/6·1/6을고정손실가중치0.5/1.5/1.5/1.5로보정한다. fallback은진위CE유지/강도손실제외.
두방법모두기존FF++val1200장×4조건FP32평균BER최소/threshold0.5로best선택, epoch0후보/guard없음/최대10/patience3.
검증native스무딩PNG만1회캐시, backbone특징은학습되므로매epoch다시추론한다. CDF는학습/선택에서읽지않는다.
CDF공식test8288장×4조건평가및기존Baseline참조포함. FP32frame FPR/FNR/BER/AUC와rawmargin저장. 테스트에는강도/정답/보조판별기불필요.
실제GPU학습은아직없다. 클래스별pivotal학습원리의적용실험이며독창성/성능확보를주장하지않는다. 기존데이터/결과/모델보존, Gitpush/Drive변경없음.
CPU4개검사통과: critic의class별gradient reversal부호/보조판별기gradient유지/fallback제외, 세모델경로의tiny실제학습과baseline보존,
정확한완료epoch중단재개, 기존Baseline참조와새평가재사용, 혼합AUC=16교차AUC평균/혼합BER=4조건평균을확인했다.
코랩7개코드셀을IPython변환후구문검사했다. 조건별평균·혼합AUC는mixture_summary.csv,4×4교차표는cross_condition_auc.csv에저장한다.

## 스무딩 캐시 최적화와 기존 Pivot 학습 호환 재개 (2026-10-16)
사용자 요청: 실행 중인 학습의 완료epoch를 살리고 속도를 개선한다. 기존trainer SHA b6ca3918f14ec866ff4746a993082c84b81ce110b0eca089a162322b3b562125만명시적으로호환허용한다.
smoothing_pixel_cache.py에native uint8 RGB bilateral 결과의무손실PNG캐시를추가했다. 강도혼합·반전·resize는그대로이며Aug/Pivot/epoch가공유한다.
기존 smoothing_roi_pilot.py와모든데이터/분할/전처리/손실/모델/optimizer/seed/validation규칙은변경하지않았다. 검증3강도생성도bilateral1회로줄이고기존PNG는재사용한다.
학습config의코드SHA외모든필드가같아야이전run재개를허용하고기존config바이트·identity를보존한다. 실행코드·캐시정보는runtime_optimization.json에별도기록한다.
last.pt의optimizer·모델·보조판별기·epoch·best·stale·history를복원한다. 완료run은skip, 미완료epoch의배치진행은저장되지않으므로해당epoch처음부터재시작한다.
--train-cache 기본/content/smoothing_pivot_bilateral_cache_v1; 처음본이미지만계산, 로컬여유2GiB미만이면새캐시저장없이계산을계속한다. 기존원본/가중치/결과삭제없음.
CPU5검사통과: 여러native크기·4강도pixel exact, 캐시hit시bilateral호출없음, paired입력/flip/fallback동일, 이전SHA체크포인트의config보존및재개후가중치/기록일치, 학습설정변경거부, 기존학습/평가재사용및7코랩셀구문검사.
실제ColabGPU속도개선량은미측정이며첫캐시생성비용은남는다. Gitpush/Drive변경/사용자Colab프로세스중단은수행하지않았다.

## 백본 고정 품질 간 진짜–가짜 순위 학습 (2026-10-16)
최근 Drive 확인에서 Pivot은 Aug를 안정적으로 능가하지 못했다. Xception best4는이미margin이커졌으며후속epoch가발산했다.
Xception은확률포화때문에AUC0.6239로보고됐지만rawmargin평균AUC는0.6543(Aug0.6477);BER는40.36%(Aug38.92%)로악화했다.
CLIP은발산없이도Pivot BER29.92%/AUC0.7599로Aug29.66%/0.7807보다낮았다. 안정성수정만으로성공할근거로해석하지않는다.
사용자승인으로scripts/ffpp_c23_smoothing_ranking.py와train_test_ffpp_c23_smoothing_ranking_seed42 MD/코랩을추가했다.
연구이유: 클래스내강도분포를억제하기보다품질이다른진짜/가짜의구분을직접학습하고,기존특징을보존하는설정에서효과를분리한다.
기존FF++baseline의encoder/classifier완전고정. 이전Current-only와같은hidden32 residual MLP를BCE/current_only와BCE+quality_ranking에서동일초기화한다.
새matched BCE조건도학습한다. source6000/epoch/기존영상균형회전표본/paired flip/원본+랜덤3강도는유지한다.
배치는source16쌍(진짜8/가짜8),32observed views. 이전Response의observed16배치와달라과거Current-only를동일대조군으로대체하지않는다.
후보는BCE+0.5mean softplus(1+max(real의두view margin)/scale-min(fake의두view margin)/scale).
scale=max(1,첫epoch관측margin절댓값q75)고정. 배치내모든진짜–가짜조합을비교하며네강도전체의최악값을보장하지않는다.
원본점수복원/teacher/target/GRL없음. LR1e-3/AdamWwd1e-4/10epoch/patience3,동일FF++val1200×4평균BER최소/epoch0후보/guard없음.
특징추출FP32/기본32장배치,매epoch12000forward를두조건공유. 기존픽셀동일native bilateral cache재사용;CDF기존스무딩PNG캐시도프로토콜일치시사용한다.
CDF공식test8288×4에서frame FPR/FNR/BER,rawmargin주AUC와확률AUC,조건혼합AUC/4×4교차AUC를저장한다. Baseline은동일forward에서획득한다.
best0이면last를별도진단으로평가한다. 새runs/manifests/evaluations root=ffpp_c23_smoothing_ranking_v1;원본데이터/과거코드·결과보존.
CPU5검사통과:최악view선택과gradient방향/두클래스배치/과거Current-only출력동일/캐시pixel exact/fallback유지/rawAUC포화처리,
세모델tiny학습·평가·완료skip/조건혼합AUC검증/전략중간중단후가중치·history완전동일재개/코랩7셀IPython문법.
전체GPU학습이나개선효과는미확인. Gitpush/Drive쓰기/새TAR생성/예약자동화없음.

## CLIP 우선 Teacher–Student UDA 비교 구현 (2026-10-16)
사용자 승인으로 scripts/ffpp_c23_smoothing_mean_teacher.py와 train_test_ffpp_c23_smoothing_mean_teacher_seed42 MD/7셀 코랩을 추가했다.
기본 CLIP B/16; Xception/DINO도 기존모델 학습범위를 재사용한다. baseline부터 독립시작하는 augmentation/mean_teacher/mean_teacher_smoothing 3조건이다.
앞선 점수 보정기/순위 손실/도메인 판별기는 사용하지 않는다. 원래 classifier와 baseline에서 정한 인코더 일부를 fine-tuning한다.
source6000/epoch·원본+무작위 스무딩·공통 seed/배치순서·FF++ val1200×4조건 평균BER 최소 선택을 맞춘다. epoch0 후보, AUC guard 없음.
선택·배포 모델은 학생이다. EMA 교사(decay0.999, FP32 eval, gradient 없음)는 임시정답만 제공하며 frozen 파라미터는 그대로 복사한다.
target CE는 confidence>=0.9인 hard pseudo-label만 전체 타깃 배치로 나누고 최대weight0.5를2epoch 선형ramp한다. 신뢰도는 정확성 보장이 아니다.
기존 dataset_inputs의 비테스트5000장과 test video/보수적filename parent제외를 재사용한다. 타깃 정답은 표본선택 전에 제거하며5:5 GT샘플러를 쓰지 않는다.
epoch6000회 영상균등 복원노출, 원본/약/중/강 각25% 관측. 교사 현재관측/학생 반대좌우반전; smoothing조건만 학생에 추가약한스무딩을 재적용한다.
깨끗한CDF대응원본이나 강도정보를 모델에 주지 않는다. 원본이미지/기존분할·preprocessing·fallback은 유지한다.
실제/가짜 teacher예측수와 임시정답 채택수/채택률/confidence/target노출계획을 저장한다. 구GT5:5 DANN과의 차이는 방법효과와 샘플링효과가 함께 달라지는 참고비교다.
최대10/patience3/min_delta0.0001, AdamW LR1e-5/weightdecay1e-4, gradientnorm1, BN통계고정, student BF16지원시BF16/아니면FP32.
source/target native bilateral PNG캐시 및 source validation변형캐시를 사용한다. backbone은학습하므로 frozen feature캐시는 사용하지 않는다.
CDF 공식test8288×4 FP32 frame FPR/FNR/BER/rawmargin AUC와 확률AUC/혼합AUC/4×4교차AUC를 저장한다. 기존 Ranking baseline raw예측이 호환되면 재사용한다.
best epoch0이면 같은baseline예측을 재사용한다. 테스트를 보고 last를best로 대체하지 않는다. 완료skip/마지막완료epoch 학생·교사·optimizer 재개 지원.
CPU4검사로 타깃label독립샘플링/임시정답마스킹/EMA/기존스무딩pixel동일/세조건학습·평가·완료skip/정확재개/기존baseline보존/7코랩셀문법을 확인했다.
실제 random-initialized CLIP B/16 구조의 forward/backward/EMA/학습범위도 검사했다. GPU 정식학습·성능개선은 미확인이다. Gitpush/Drive쓰기/새TAR/예약없음.

## Teacher–Student UDA 완료 결과와 Soft 정답 비교 구현 (2026-10-17)
앞 문단의 GPU 미확인은 코드 작성 시점의 기록이다. 이후 Drive에서 CLIP/Xception의 Aug·일반 Mean Teacher·추가 스무딩 Mean Teacher 총6개 완료 run을 확인했다.
4조건 평균 BER/AUC: Xception baseline38.56%/0.6731, matched Aug38.92%/0.6477, 일반UDA33.76%/0.7060.
CLIP baseline32.07%/0.7997, matched Aug29.66%/0.7807, 일반UDA31.06%/0.8015. 단일seed 개발 평가이며 보편적인 개선으로 주장하지 않는다.
사용자가 soft 정답 및 CDF weight0.5·0.25 두 조건 구현을 승인했다. scripts/ffpp_c23_smoothing_mean_teacher_soft.py와 동명 train_test_*_seed42 MD/7셀 코랩을 추가했다.
기존 hard Mean Teacher 엔진 파일을 변경하지 않고 학습/EMA/학생 선택/평가/캐시/재개 함수를 재사용한다. 새 protocol은 ffpp_c23_smoothing_mean_teacher_soft_v1이다.
두 soft 조건은 FF++ baseline에서 각각 독립시작한다. 교사 확신>=0.9 mask를 유지하고 T=1 교사 확률 그대로 soft CE를 계산하며 전체 target 배치 크기로 나눈다.
CDF 최대 weight0.5 또는0.25만 바꾸고 기존2epoch ramp를 유지한다. source CE·FF++6000/epoch·CDF pool5000 및6000영상균등노출·원본/약/중/강25%·반대flip·EMA0.999·학습범위·split·native ROI·fallback은 기존 그대로다.
CDF 실제 정답/GT5:5샘플링/추가약한스무딩은 사용하지 않는다. 기존 hard의 고정 validation/target CSV와 현재 목록·순서·데이터해시·예산을 비교한다.
FF++ val1200×4 mean BER 최소·epoch0후보·guard없음·최대10/patience3를 유지한다. 선택·추론은 학생 best.pt이다.
신규 checkpoint에는 재사용 엔진 해시와 soft 확장파일 해시를 구분해 저장한다. 기존 native pixel 캐시 identity 및 기존 hard checkpoint/code는 보존한다.
기본 CLIP/Xception을 제공하며 MODELS를 하나만 남겨 모델별 코랩으로 나눌 수 있다. 새두조건만 학습/추론하고 이전 Baseline/Matched Aug/Hard UDA는 FP32 raw prediction·manifest·평가프로토콜이 맞는 저장 결과를 읽는다.
comparison_with_hard.csv, condition_means_with_hard.csv, deltas_vs_hard.csv 및 기존 혼합/교차AUC·조건별원시예측을 저장한다. hard 결과 누락 시 새추론 전에 중단한다.
CPU 작은fixture로 soft CE gradient/teacher detach/신뢰도mask/전체배치분모, hard와동일source·target계획, 두weight, 완료skip, 정확한학생·교사재개, 원본hard보존, 비0epoch학생추론·baseline재사용·비호환참조거부·코랩7셀구문을 검사했다.
이 soft 조건의 전체GPU학습·성능은 아직 확인하지 않았다. 데이터/Drive 변경, 새TAR 생성, Gitpush, 자동예약은 수행하지 않았다.

## 교사·학생 CDF 입력 조건 분리 실험 구현 (2026-10-17)
앞선 문단의 soft GPU 미확인은 작성 시점의 기록이다. 이후 Drive 결과 확인에서 Xception/CLIP soft0.5·0.25와 DINO hard UDA·학생 추가스무딩·matched Aug 완료를 확인했다. DINO 4조건 평균 BER/AUC는 Aug31.34%/0.7416, hard UDA33.70%/0.7305이며, 교사의 높은 확신이 실제정확도라는 뜻은 아니다.
사용자가 DINO 두 조건 추가를 승인했다. ffpp_c23_smoothing_teacher_views.py와 train_test_ffpp_c23_smoothing_teacher_views_seed42 MD/7셀 코랩을 추가했다. 기본 MODELS=['dinov2_b14']; 세 모델을 지원한다.
normal_only는 교사·학생 모두 스무딩 전 CDF 크롭, clean_to_mixed는 교사 원본·학생 원본/약/중/강 각25%다. 두 조건 모두 기존 FF++ baseline부터 독립 시작하고, 적응 시 비라벨 스무딩 전 target 원본을 확보할 수 있다는 가정을 명시한다. 이미 보정된 이미지에 원본 짝이 없는 설정과 다르다.
기존 hard EMA 엔진과 soft 파일은 변경하지 않고 학습·평가·manifest 참조 함수를 재사용한다. source Aug, FF++6000/epoch·검증1200×4·CDF pool5000/6000노출, seed42, hard confidence0.9, targetweight0.5/2epoch ramp, EMA0.999, LR1e-5, BN고정, 최대10/patience3, 학생 meanBER 선택/epoch0 후보를 유지한다. 실제 CDF 정답과 클래스가중치/안정성 선별은 쓰지 않는다.
기존 hard와 source검증·target 고정 목록/순서·해시를 비교한다. legacy target plan을 먼저 모두 추첨해 이미지경로·영상·flip을 보존하고, condition은 실제학생조건으로 기록한다. scheduled_condition은 legacy 무작위추첨, teacher_condition은 normal이다. 조건별 manifest 하위폴더로 노출 CSV 충돌을 방지한다. 기존 native pixel/validation 캐시는 재사용한다.
새 출력은 ffpp_c23_smoothing_teacher_views_v1. CDF test8288×4에서 두 모델만 새 추론하고, 기존 baseline/matched Aug/hard UDA/학생 추가스무딩 UDA는 저장예측을 검증해 재사용한다. 원본·4조건평균·강-원본 FPR/FNR/BER/AUC 변화·pooled AUC와 기존 Aug/hard 대비 차이를 domain_and_smoothing.csv로 저장한다. 테스트 threshold/epoch선택이나 인과적 효과 분리 완료로 주장하지 않는다.
CPU 작은fixture 4검사로 원본교사/학생변형의 정확한 픽셀·fallback·반대flip, target label 독립노출, hard손실 재사용, 기존membership/결과보존, 새 학습/완료skip/정확한 학생·EMA 재개/3모델 경로, 기존결과 GPU재추론 없음, 원본/열화 분리집계, 코랩7셀/MD일치·직접진행률명령을 확인했다. 실제 GPU 학습과 성능 개선은 미확인이다. 새TAR/Drive쓰기/Gitpush/예약은 수행하지 않았다.

## 저장 예측 기반 FPR·FNR 동시 개선 가능성 진단 (2026-10-17)
사용자 요청으로 scripts/analyze_smoothing_error_tradeoff.py와 analyze_smoothing_error_tradeoff_seed42 MD/4셀 코랩을 추가했다.
학습 엔진·Torch·모델·이미지/TAR를 불러오지 않는 CPU 분석이다. 기존 hard/soft/교사 원본 평가 폴더에서 FP32 raw-margin CSV/JSON을 읽는다.
세 모델 Baseline/Matched Aug/기존 일반 UDA는 필수, 추가 스무딩/Soft/normal_only/clean_to_mixed는 4조건 완료 결과만 포함한다. 동일 checkpoint의 중복 사본은 metadata와 점수를 대조하고, 충돌은 중단한다.
동일 이미지·정답·영상·ROI·manifest·스무딩 설정과 원래 지표를 확인한다. 원본 파일은 수정하지 않고 입력 해시·패키지 버전·누락 조건을 별도 저장한다.
기존 margin>=0 성능과, 각 기준(Baseline/Matched Aug)의 FN 개수 이하에서 최소 FP·FP 개수 이하에서 최소 FN을 계산한다. 동점은 하나의 그룹으로 처리하고 보간하지 않는다.
조건별 oracle와 별도로 4조건 전체에 하나의 공통 cutoff를 적용해 각 조건의 두 오류를 동시에 유지·감소시킬 수 있는지 검사한다. 엄격 감소는 같은 이미지에서 FP와 FN 각각 최소1장 감소다.
CDF 테스트 정답을 사용한 가능성 진단이며 배포 임계값·정식 성능·모델 선택으로 사용하지 않는다. 관측 AUC 유지 여부는 통계적 동등성 보장이 아니다. 영상 내 프레임 상관을 무시한 유의성 주장은 하지 않는다.
출력 fixed_threshold.csv/fixed_condition_means.csv/operating_points.csv/one_threshold_all_conditions.csv/analysis_protocol.json 및 모델별 오류 곡선 PNG. 기존 결과는 보존하고 빈 시간별 diagnostics 폴더만 사용한다.
작은 CPU fixture로 동점·상수 점수·전체 결정 임계값·단순 점수 이동, 조건별 성공과 공통 기준 실패의 차이, CSV 순서·중복·부분 완료·충돌·원본 보존 및 코랩4셀/MD 일치를 확인했다. 실제 전체 Drive 예측 분석은 아직 실행하지 않았다. Gitpush/Drive변경/새TAR/예약은 수행하지 않았다.

## 스무딩 판단 일치 기반 임시 정답 선택 UDA 구현 (2026-10-17)
사용자가 앞선 분석 후 제안한 Xception 우선 비교를 승인하여 scripts/ffpp_c23_smoothing_stable_uda.py 및 train_test_ffpp_c23_smoothing_stable_uda_seed42 MD/코랩7셀을 추가했다.
기존 mean_teacher.py 엔진 파일을 수정하지 않고 학습·검증·재개·평가를 재사용한다. 기본 Xception이며 MODELS로 CLIP/DINO도 지원한다. 기존 FF++ baseline부터 stable_agreement와 random_agreement_count를 독립 학습한다.
교사는 관측 view를 포함한 원본/약/중/강 네 view를 동일 flip으로 추론한다. 관측 view confidence>=0.9이며 네 argmax가 모두 같으면 stable 채택한다. 네 view 모두 confidence0.9를 요구하지 않는다. 학생은 기존 관측 view의 반대flip 한 장만 본다.
스무딩 전 비라벨 target crop을 사용할 수 있다는 가정이며 테스트에서는 학생 한 번만 추론한다. fallback은 네 view 동일 입력으로 처리하고 실제 정답/강도를 모델 입력에 사용하지 않는다. 일관된 오답도 통과할 수 있다.
무작위 대조군은 자기 EMA 교사의 confidence 후보 중 stable 가상 mask와 같은 수를 매 배치·예측 클래스별로 추출한다. RNG를 분리한다. 독립적인 두 run의 EMA가 달라지므로 두 run의 전체 채택 수/예측 클래스 구성까지 일치한다고 주장하지 않는다. 배치 CSV 및 history에 candidate/agreement/selected 수와 mask hash를 저장한다.
source6000/epoch, 기존 고정 target5000장·영상균등6000노출/epoch 및 강도25% 계획을 재사용한다. 기존 hard manifest와 데이터/validation membership을 확인한다. source 원본+random smoothing CE, target hard CE를 전체 타깃 배치로 나누는 규칙, 가중치0.5/2epoch ramp/EMA0.999/LR1e-5/BN통계고정 유지.
최대10epoch/patience3, FF++ validation1200장×4조건 평균BER 최소, epoch0 후보/AUC guard 없음. 네 teacher view 때문에 기존 UDA보다 계산량이 많으며 teacher-batch-size32로 메모리를 제한한다. expensive native bilateral은 이미지당 한번 계산/읽고 강도별blend에 재사용한다. 기존 target/source/cache identity를 유지한다.
CDF 공식test8288장×4에서 새 두 학생만 평가하고 기존 Baseline·Matched Aug·Hard UDA의 저장 예측을 검증하여 병합한다. fixed cutoff0.5의 FPR/FNR/BER/raw-margin AUC, 평균/조건별 차이와 두 오류 동시 감소·수치상 AUC 유지 flag를 저장한다. CDF test로 모델/임계값을 선택하지 않는다.
새 출력 roots ffpp_c23_smoothing_stable_uda_v1, 기존 이미지·manifest·체크포인트·평가·기존 엔진 코드 보존. 새 TAR/ZIP 없음. 코랩은 직접 !python-u 진행률을 표시하고 완료skip/마지막 완료epoch 재개를 지원한다.
작은 CPU 5검사로 native pixel·observed-first 순서·fallback·label-free 계획, 안정성mask/신뢰도/배치·예측클래스별 무작위quota/빈mask·gradient·전체배치분모·RNG보존, teacher chunk/EMA, 세 모델 tiny 학습/완료skip/동일가중치 재개, 기존 평가 재추론 없음·새8조건 추론/비교표·재사용·기존파일보존, 코랩7셀/MD일치·IPython문법을 확인했다. 실제 전체 GPU 학습이나 성능 개선은 아직 확인하지 않았다. Gitpush/Drive쓰기/예약 없음.

## UDA 오류 변화 및 최종 EMA 임시 정답 진단 구현 (2026-10-18)
사용자 요청으로 analyze_smoothing_uda_failures.py, diagnose_smoothing_pseudo_labels.py 및 analyze_smoothing_uda_failures_seed42 MD/8셀 코랩을 추가했다.
첫 4셀은 기존 FP32 raw-margin CSV/JSON만 읽는 CPU 분석이다. 기본 Xception의 Baseline·일반 Mean Teacher·stable_agreement 4조건을 crop_path로 연결한다.
클래스별 기존 오류/고친 오류/새 오류/순변화를 집계하고, baseline 원본정답→스무딩오답 부분집합의 구제와 후보 원본정답 유지 여부를 따로 저장한다.
기존 오류-reader의 실제 점수·지표·manifest·ROI·스무딩·checkpoint 일치 검사를 재사용한다. 기존 reader/trainer/hash를 바꾸지 않는다.
후반 4셀은 선택적 GPU 진단이다. 기존 고정 비라벨 target5000장에 비테스트 원본 metadata의 정답을 추론 후 집계용으로만 복원한다.
일반UDA와 판단일치UDA의 last.pt 최종 EMA를 strict 복원해 각 이미지의 원본/약/중/강을 무반전 FP32 추론한다. 기본2교사×5000×4=40000입력, 재학습 없음.
best.pt는 선택된 학생이므로 교사로 쓰지 않는다. 과거 step별 교사를 저장하지 않았으므로 이 진단은 학습 당시 임시정답의 재생이 아닌 최종 snapshot 사후평가다.
두 교사의 마지막 epoch/선택 학생epoch를 기록하며, 원인 증명·테스트 기반 model/threshold 선택으로 사용하지 않는다.
타깃 정답은 추론 Dataset/모델에 전달하지 않는다. 고정 pool과 테스트의 영상/이미지 중복·기존 보수적 제외목록·비테스트 manifest hash를 확인한다.
기존 non-test real/fake TAR 두 개에서 고정 target crop/metadata만 준비하며 FF++/test TAR/baseline 가중치는 필요 없다. 기존 native bilateral cache identity를 그대로 재사용한다.
임시정답 품질은 실제/예측클래스·confidence gate/추가agreement gate별로 보고, pool 이미지평균과 영상균등 기대가중치를 구분한다. 실제 과거 채택수 재현이나 GT5:5샘플링이 아니다.
새 출력은 diagnostics/smoothing_uda_failure_v1/seed42_시간/errors 및 pseudo. 기존 데이터·학습설정·평가·체크포인트를 변경하지 않고, 새 TAR/ZIP·Gitpush·Drive쓰기·예약은 수행하지 않았다.
작은 CPU6검사로 오류집계 항등식/스무딩 paired 부분집합, 순서 다른 예측 연결·원본보존·충돌거부, label-free frozen membership/누출거부,
필요 TAR member만 추출·미요청이미지 제외·재사용, 최종EMA/래핑key복원·tiny모델추론·완료결과재사용·snapshot변경거부, MD/코랩8셀 일치·IPython문법을 확인했다.
이 진단의 실제 전체 CDF/GPU 실행은 아직 수행하지 않았다. 앞선 GPU 미확인 문장은 각 구현 시점의 상태이며 실제 실험 완료 여부는 해당 run의 결과 파일로 확인한다.

## 임시 정답 진단의 중복 Drive 폴더 경로 수정 (2026-10-18)
사용자 Cell7 FileNotFoundError를 확인했다. Drive runs 아래 ffpp_c23_smoothing_mean_teacher_v1 이름의 폴더가 두 개이며,
Xception은1M8_b94wg28NBMpHMfL9_LTmDNLhfuLiF, CLIP/DINO는1nBtrAyC1Duk4Rn5t03ARErgv3QJV8zP- 아래 저장된 것을 읽기전용 목록으로 확인했다.
Xception 일반UDA run1WNxM9vAh02RBXIY7mzjOxSe_QYMaEFP-의 config/completed/last333738717bytes,
stable run1304Z4iSJpDUF9PVw3goELzpl7JdjKinL의 config/completed/last333737693bytes 존재를 확인했다. 학습파일 소실이 아닌 경로 문제다.
진단 helper에 run root/하위폴더/같은 runs내 경로 검색과 모호한 복수 매치 거부를 추가했다.
마운트 이름으로 접근하지 못하면 명시한 Drive run ID에서 config/completed/last만 read-only API로 /content 전용 캐시에 읽는다.
정확한 폴더명·필수파일 유일성·provider MD5/크기를 확인하고 원본 Drive 파일/폴더와 학습 engine은 변경하지 않는다. 임시 다운로드만 정리한다.
Cell7에 코드갱신·필요시 Colab Google인증·검증된 Xception 두 ID 전달을 넣어 1–6셀/데이터 준비 없이 재시도할 수 있게 했다.
두 run 위치를 추가 추론 전에 먼저 확인한다. 작은CPU8검사(기존6+중첩/중복경로 fallback·read-only API fixture 및 checksum cache)와 MD/8셀 문법이 통과했다.
실제 Colab 사용자 인증/API 다운로드/GPU 진단 성공은 아직 미확인이다. Gitpush/Drive쓰기/폴더 재구성은 수행하지 않았다.

## 일반 TS에 source 원본–스무딩 점수 학습 추가 (2026-10-18)
사용자가 기존 일반 Teacher–Student UDA에 FF++ paired 점수 차이 항목 하나를 추가하는 비교를 승인했다.
scripts/ffpp_c23_smoothing_mean_teacher_pair.py와 train_test_ffpp_c23_smoothing_mean_teacher_pair_seed42 MD/7셀 코랩을 추가했다.
기본 Xception이며 CLIP/DINO도 MODELS로 지원한다. 새 mean_teacher_pair만 원래 FF++ baseline에서 독립 시작하고 기존 Baseline/Matched Aug/mean_teacher는 저장 FP32 raw-margin 예측으로 비교한다.
추가 항목은 0.1 * mean[detected * SmoothL1((fake-real)_smooth - (fake-real)_clean,0)], beta=1 raw-logit 단위다.
양쪽 현재 학생 점수 모두 역전파하며 detach/과거 baseline 점수 복원/새 보정기 없이 원래 인코더 허용 범위와 분류기를 학습한다.
fallback은 CE만 적용하고 분모는 전체 source pair 수다. 가중치0.1은 성능 확인 전 고정한 탐색값이며 최적값으로 주장하지 않는다.
기존 source6000/epoch 원본+랜덤3강도 CE, 비라벨 CDF 고정5000/영상균등6000노출/4조건25%, confidence0.9 hard CE,
targetweight0.5/2epoch ramp/EMA0.999/LR1e-5/BN통계고정/validation1200x4 평균BER/epoch0/최대10/patience3를 유지한다.
기존 source logits를 재사용하여 추가 source/target forward는 없다. 원래 teacher 엔진과 pixel/validation 캐시 identity는 보존한다.
새 pair_v1 run/config/history/best/last/manifest/evaluation에 별도 저장하고 완료skip/완료epoch 재개를 지원한다.
CDF test8288x4 신규 후보만 추론, 기존 세 조건 저장결과를 확인하여 comparison_with_hard/condition_means_with_hard/deltas_vs_hard/smoothing_changes.csv를 저장한다.
CPU3검사로 손실/두 view gradient/fallback/점수 공통 offset 불변성, weight0의 기존 hard TS 가중치 완전 일치,
동일 source/target 계획, 양의 가중치 학습·완료skip·중단재개 가중치 일치·기존 결과 재사용·기존 파일 보존,
7코드셀 MD/노트북 일치 및 IPython 문법을 확인했다. 전체 GPU 성능은 미확인이다. Gitpush/Drive쓰기/새TAR/예약 없음.

## Source worst-view Aug/TS 비교 구현 (2026-10-19)
앞선 paired Huber GPU 실험은 이후 Drive 결과에서 Xception/DINO 완료를 확인했다.
4조건 평균 BER/AUC는 Xception 기존 TS33.76%/0.7060 대비 pair35.06%/0.6966,
DINO 기존 TS33.70%/0.7305 대비 pair35.60%/0.6749로 악화했다. 변화 폭 감소만으로 성능 개선으로 해석하지 않는다.
사용자가 다음 비교를 승인하여 scripts/ffpp_c23_smoothing_worst_view.py 및 train_test_ffpp_c23_smoothing_worst_view_seed42 MD/7셀 코랩을 추가했다.
기본 Xception, MODELS로 CLIP/DINO도 지원한다. worst_view_aug/worst_view_ts를 원래 FF++ baseline에서 독립 시작한다.
FF++ 손실만 이미지별 max(CE(clean,y),CE(random smoothing,y)) 후 평균으로 변경한다. 배치 전체 최대가 아니다.
정확한 tie는 양쪽 gradient 반씩, 별도 배율/Huber/점수 복원/보정기 없이 원래 인코더 허용범위와 classifier를 학습한다.
최대값은 평균보다 크거나 같아 고정 targetweight 아래 상대적인 source/target 영향까지 같다고 주장하지 않는다.
기존 paired 두 source forward를 재사용하며 추가 forward는 없다. source6000/epoch, FF++ val1200x4 평균BER 최소,
epoch0 후보/최대10/patience3/LR1e-5/BN통계고정/전처리/데이터/캐시를 유지한다.
TS만 기존 label-free CDF5000/영상균등6000노출/4조건25%/반대flip/confidence0.9 hard CE/weight0.5 ramp2/EMA0.999를 적용한다.
Aug는 target forward/teacher/임시정답 손실이 없고 target 노출과 가중치는 history에0으로 기록한다.
CDF 공식test8288x4 신규 두 조건만 평가, 기존 Baseline/Matched Aug/일반TS 저장예측을 검증해 재사용한다.
comparison_with_reference/condition_means_with_reference/smoothing_changes/deltas_vs_matched.csv에 각각 기존 Aug/TS 대비 차이를 저장한다.
완료skip/완료epoch 학생·교사·optimizer 재개/직접 진행률을 지원하고 새 worst_view_v1 경로로 과거 엔진·데이터·결과를 보존한다.
CPU3검사로 이미지별 선택·진짜/가짜 정답방향 gradient·tie, 3모델 tiny 학습·Aug target학습없음·동일source/target계획,
완료skip·재개가중치 일치·기존결과 재사용·기존파일 보존·7셀 MD/노트북 일치·IPython 문법을 확인했다.
전체 GPU 성능은 미확인이다. Gitpush/Drive쓰기/새TAR/ZIP/예약 없음.

## CLIP Worst-view 결과 및 공간 전처리 확인 (2026-10-19)
후속 Drive 읽기에서 CLIP Worst-view Aug(6epoch/best3)·TS(4epoch/best1) 완료를 확인했다.
4조건 평균 BER/AUC: 기존 Aug29.6564%/0.780697 대비 Worst Aug29.1756%/0.781749,
기존 TS31.0593%/0.801455 대비 Worst TS32.5732%/0.785056. Aug의 원본→강한 FPR 증가는5.72→8.04%p로 오히려 커졌다.
마지막 TS 평가가 요약표의 Aug행을 덮어써 조건별 Aug JSON4개를 직접 합산했다. 새 실험에서는 완료된 다른 조건도 집계한다.
로컬 crop_manifest 전체에서 FF++80,000장/CDF104,449장 모두 정사각형(비정사각0)을 확인했다.
실제24개RGB PNG에서 CLIP Resize224+CenterCrop224와 DINO direct224의 정규화 전 픽셀도 동일했다.
현재 ROI에서는 center crop으로 얼굴 가장자리가 추가 제거된다는 우려가 적용되지 않는다. 기존 학습·평가 전처리를 유지하며 재학습하지 않는다.

## 스무딩 특징 변화 전이 학습 구현 (2026-10-19)
사용자가 AUC 개선을 핵심 목표로 재설정하고 새 방법의 코드/코랩을 요청했다.
ffpp_c23_smoothing_delta_transfer.py 및 train_test_ffpp_c23_smoothing_delta_transfer_seed42 MD/7셀 코랩을 추가했다.
기본 CLIP, MODELS로Xception/DINO도 지원한다. 원래FF++baseline에서 ce_rank/source_delta/target_delta를 독립 시작한다.
공통 source6,000/epoch·진짜3,000/가짜각750·원본+무작위스무딩·같은flip, 세새조건모두배치1:1을보장한다.
source/target donor는 원본/약/중/강 관측→추가약한스무딩, 같은flip 두 이미지의 EMA인코더 특징 차이를 계산한다.
source donor는같은sourceepoch순서만섞고, target donor는기존CDF5000장 GT제거후영상균등6000회노출이다.
CDF실제정답/임시정답/confidence/강도입력없이 delta만사용, EMA0.999는특징측정용이며delta역전파없음.
delta를source원본·스무딩특징에양의방향으로더하고 recipient norm최대5%로cap, 절대증폭없음/첫2epoch선형ramp.
변화가 조작 단서를 보존한다는 것은실험가정이다. donor도메인별내용·classprior차이가있으므로순수스무딩스타일인과효과로단정하지않는다.
실제source정답CE +0.25모든진짜–가짜·view쌍softplus(real_margin-fake_margin), 인코더기존범위와분류기함께학습.
세새조건의선택기준을 FF++val1200×4 평균raw-margin AUC최대로변경, 정확한AUC동률이면BER최소.
epoch0후보/최대10/patience3/min_delta0.0001/BN통계고정/LR1e-5/AdamW. 기존Aug·TS의BER선택은참고비교로명시한다.
nativeROI·스무딩·모델별전처리·기존split/targetpool/test제외유지, 기존native bilateral/probe/validation캐시재사용.
CDF공식test8288×4 frame FP32평가까지포함하며추론은학생모델1회, donor/EMA/짝이미지불필요.
기존Baseline/Aug/일반TS는저장예측만재사용, 새comparison/means/deltas_vs_controls/smoothing_changes와선택기준차이를저장한다.
완료skip/완료epoch학생·EMA·optimizer정확재개/완료된형제조건집계/직접진행률지원. 추가donor2viewFP32추론으로계산량이대조군보다크다.
CPU4검사로normcap·증폭금지·donor gradient차단·source gradient·GT순위방향·0delta중복view목적일치,
동일flip/픽셀/fallback·targetlabel독립·3모델tiny학습/완료skip/정확재개/선택기준·저장결과재사용·기존파일보존·7셀문법을확인했다.
전체GPU학습·성능·독창성은미확인이다. Gitpush/Drive쓰기/데이터수정/새TAR/ZIP/예약없음.

## CE+Rank에 CDF 분류 감독 추가 비교 (2026-10-20)
이후 Drive 결과에서 delta-transfer 3모델 완료를 확인했다. 4조건평균 raw-margin AUC는
CE+Rank Xception0.6618/CLIP0.8161/DINO0.7579, target_delta0.6619/0.8087/0.7455로 타깃delta의 추가효과를 확인하지 못했다.
사용자 승인으로 ffpp_c23_smoothing_rank_adaptation.py 및 train_test_ffpp_c23_smoothing_rank_adaptation_seed42 MD/코랩을 추가한다.
기본DINO, MODELS로다른2백본지원. 기존CE+Rank 완료checkpoint·저장예측은같은데이터/설정/코드/초기값인지확인후읽기전용재사용한다.
새rank_ts/rank_sda는각각원래FF++baseline에서독립시작하며, CE+Rank가중치에서이어학습하지않는다.
공통source6000/epoch·GT CE+0.25rank·1:1배치·원본+랜덤스무딩·동일flip·원래인코더범위+분류기·LR1e-5/BN고정유지.
기존비테스트CDF5000장을GT제거후영상균등6000회/epoch·4조건25%·학생반대flip으로동일계획노출한다.
rank_ts는EMA0.999/confidence0.9 hard CE, rank_sda는비테스트metadata의실제정답CE를전체관측이미지에적용한다.
두target손실모두전체배치분모·최대weight0.5/2epoch ramp. SDA는UDA가아니며confidence선별유무에따른감독량차이도있어정답오류만의인과효과나엄밀한상한으로주장하지않는다.
source validation1200×4 평균raw-margin AUC최대·동률BER최소·epoch0후보·최대10/patience3/min_delta0.0001을기존CE+Rank와맞춘다.
같은예산상한이어도조기종료epoch와target계산량은다를수있다. CDF test는모델선택에사용하지않는다.
기존pixel/validation캐시재사용·진행률·완료skip·완료epoch학생/EMA/optimizer재개·새버전출력보존을지원한다.
CDF test8288×4 신규두조건만FP32 frame평가,기존CE+Rank와Baseline예측재사용,조건별/평균/오류변화/CE+Rank대비차이를저장한다.
GPU정식학습·성능은아직미확인이다. 데이터/과거run/Drive변경·새TAR·Gitpush·예약없음.
작은CPU3검사로GT독립계획·TS/SDA손실과gradient·라벨metadata검사,기존CE와동일source배치,
targetweight0일때CE+Rank와학생가중치완전일치·TS경로GT복원미호출·정확한학생/EMA재개·완료skip,
선택AUC·학습된학생평가·기존CE예측/파일보존·평가재사용·7셀MD/코랩일치와IPython문법을확인했다.

## 0·1단계: pooled AUC 선택 CE+Rank와 pool v2 균형 SDA (2026-10-21)
Drive 저장 예측 재분석: 기존 CDF 적응 풀은 전이적 filename 제외로 비테스트 가짜 5,299→74영상(22장면), Celeb-real 482→38, YouTube-real 230 유지.
Celeb-real/YouTube-real별 FPR·AUC는 거의 같아 YouTube 지름길 가설은 지지되지 않음. SDA 미탐은 가중치 없는 78:22 target CE와 일치.
4조건 평균 AUC는 조건 간 순위 붕괴(강 진짜 vs 원본 가짜 CLIP0.682/DINO0.609/Xception0.459)를 반영하지 못해 선택 기준을 pooled AUC로 바꾼다.
사용자 요청으로 scripts/ffpp_c23_smoothing_rank_sda_pool_v2.py, tests/test_smoothing_rank_sda_pool_v2.py, train_test_ffpp_c23_smoothing_rank_sda_pool_v2_seed42 MD/7셀 코랩을 새로 추가했다. 기존 파일은 변경하지 않았다.
pool v2: test 장면키(두 ID 모두) 1-hop 제외(비전이), 가짜 장면당 최대3, 진짜 수/CR:YT는 공식 test 영상비, 장면키 연결 묶음 단위 80/20 adapt-train/adapt-val.
로컬 매니페스트 재현: adapt-train CR110/YT72/가짜355영상, adapt-val 28/18/81영상. 구성에 폴더·라벨 metadata를 쓰는 설계 가정이며, UDA용 adapt_train_unlabelled.csv는 라벨 없음.
조건: ce_rank_pooled(CE+Rank v1과 동일 학습·선택만 변경), sda_new_balanced(새 풀 실제정답, 배치 정확히1:1), sda_old_balanced(기존 풀 1:1, 선택).
best.pt=FF++ val4조건 pooled AUC 최대(동률 최소 조건간AUC), best_adapt_val.pt=CDF adapt-val 동일규칙 진단 선택. 조기종료는 FF++ 기준. 공식 test는 선택에 쓰지 않음.
평가: 공식test8288×4, pooled/조건간AUC, CR/YT별 FPR·AUC, 선택 epoch, CE+Rank(pooled) 대비 영상 부트스트랩(테스트 표본 변동만).
CPU fixture 4검사 통과: 1-hop 제외/장면 cap/묶음 분할/고정 재현, 배치1:1 계획, pooled·교차AUC=sklearn, ce_rank_pooled가 CE+Rank v1 가중치와 완전 일치, 중단 재개 일치, 평가 재사용, 코랩7셀 문법.
GPU 학습·성능은 미확인. 장면키는 filename 후보 관계이며 source/driver 검증이 아님. Git push/Drive 쓰기 없음.

## pool v2 동시 실행 중복 Drive 폴더 사고와 대응 (2026-10-21)
CLIP·DINO 코랩을 3초 간격(15:38:14/15:38:17)으로 시작해 runs/·manifests/ 아래 ffpp_c23_smoothing_rank_sda_pool_v2가 각각 2개 생겼다.
runs A=1IXoSrb8q01qrm6kbCn4kJ0Xi0w0J1vRa(첫 CLIP 실행, sda_old 미완료), B=1jjSyOs4SArqc1ddVs4q6xlA5RY4OrZwM(DINO + 19:20 재시작 CLIP). manifests A=1ZrXsAj5-MfpP6bxKYfC3u4xLqUzMG63E, B=1nAauRfJd5ELz6tJ-zdP9N8s5b0zvdGuH.
CLIP 재시작이 마운트에서 B를 잡아 CE+Rank부터 재학습했다(약 2시간40분 손실). A·B의 CLIP ce_rank_pooled/sda_new_balanced 파일 크기는 바이트 단위로 같아 결정적 재현으로 판단(내용 해시 비교는 안 함).
대응: scripts/drive_folder_guard.py(읽기 전용 Drive API, 결과 경로 각 단계에 같은 이름 폴더가 2개 이상이면 중단)를 추가하고 코랩 Cell 2에서 실행한다.
평가 출력은 모델별 최상위 폴더 evaluations/ffpp_c23_smoothing_rank_sda_pool_v2_<모델>/seed42로 바꿔 동시 Cell 6의 경쟁 생성을 없앴다. runs/manifests 경로는 진행 중 run 재개를 위해 유지한다.
같은 날 사용자 요청으로 runs·manifests의 A 두 폴더 이름 끝에 _duplicate_clip_first_run을 붙였다(삭제 아님). 이제 원래 이름은 B만 남는다. 앞으로 새 실험의 첫 새 폴더 단계는 모델별 이름으로 만든다.

## CLIP pool v2 결과와 pool v2 UDA 구현 (2026-10-22)
CLIP 0·1단계(Drive 확인, seed42): 4조건 pooled AUC Baseline0.796 / CE+Rank(pooled, epoch1)0.814 / SDA새풀0.880(epoch2) / SDA새풀@adapt-val0.891(epoch4) / SDA기존풀1:1 0.798.
SDA새풀 vs CE+Rank 영상 부트스트랩 +0.066[+0.046,+0.088]. 평균 FPR/FNR 20.1%/20.1%, 원본→강 FPR증가 +8.1%p(CE+Rank +15.8%p).
기존풀1:1은 FNR63%로 미탐이 남아, 예전 SDA 미탐은 클래스 비율보다 좁은 가짜 다양성(74영상) 때문으로 해석. FF++ val이 정체되어 epoch4에서 조기종료됐고 adapt-val은 계속 상승.
인물 겹침: adapt-train에 나온 인물의 test 영상에서 SDA-CE+Rank +0.072[+0.043,+0.098], 안 나온 인물(진짜17/가짜24영상)에서 -0.026[-0.111,+0.054]. 인물 익히기와 도메인 적응을 구분하지 못한 한계로 명시.
사용자 결정: 보조 인물분리 실험은 미루고 UDA 진행. scripts/ffpp_c23_smoothing_rank_uda_pool_v2.py, tests/test_smoothing_rank_uda_pool_v2.py, train_test_ffpp_c23_smoothing_rank_uda_pool_v2_seed42 MD/7셀 코랩을 추가했다.
ts_confidence(EMA 확신0.9 hard, 예전 TS 규칙)와 ts_rank_quantile(epoch 시작 EMA 교사 margin 상위20%=가짜/하위20%=진짜, 비율 추정 없음). BBSE는 FF++→CDF 점수 이동으로 가정이 깨져 학습에 쓰지 않고 기록만 한다.
pool v2 adapt-train을 라벨 없이(3열) 사용. 라벨은 임시정답 정확도·BBSE 오차 기록용으로만 읽으며, 라벨을 뒤집어도 가중치가 같음을 테스트로 확인. target weight0이면 pool v2 CE+Rank 가중치와 완전 일치.
선택 FF++ val pooled AUC, adapt-val은 기록만. 평가에 간격 회복률(방법-CE+Rank)/(SDA-CE+Rank), 나온/안 나온 인물별 pooled AUC, 부트스트랩 포함. CE+Rank·SDA는 pool v2 저장 예측을 읽기만 한다.
동시 실행 규칙 적용: runs/manifests/evaluations 첫 새 폴더가 모델별(<실험>_<모델>), Cell2 Drive 중복 검사, START NEW/RESUME 표시. CPU 테스트 4개(+기존 pool v2 5개) 통과. GPU 학습은 미실행.

## pool v2 UDA 결과와 교사 보호 실험 구현 (2026-10-22)
CLIP/DINO UDA(seed42, FF++ 조기종료): 4조건 pooled AUC CE+Rank 0.814/0.723, TS 확신0.9 0.823/0.725, TS 순위20% 0.778/0.678, SDA 0.880/0.858.
TS 확신 vs CE+Rank CLIP +0.010[+0.001,+0.017], DINO +0.002[-0.011,+0.015]; TS 순위는 -0.035/-0.045로 구간이 0 아래. 간격 회복률 TS 확신 14%/1%.
순위 규칙은 가짜 쪽 임시정답 오답 6~8%로 깨끗했지만 진짜 쪽을 가짜 쪽과 같은 수만큼 채택해 진짜 라벨의 24~37%가 실제 가짜였고 FNR이 크게 증가했다. 전체 정확도는 확신 규칙과 비슷(74~84%).
DINO 순위 규칙은 epoch이 지날수록 진짜 쪽 오답 증가·adapt-val 하락(악순환). TS 확신은 두 모델 모두 epoch1이 선택됐지만 기록용 adapt-val은 이후 계속 상승.
DINO CE+Rank는 동일 데이터 순서에서도 GPU 비결정성으로 이전 run과 epoch1부터 갈라져 CDF 0.757 vs 0.723; DINO의 0.03~0.04 차이는 run 변동으로 본다.
사용자 결정: 순위 규칙 중단, source 손실은 CE+Rank 유지(고정 source 설정으로 보고), 조기종료 대신 고정 6epoch(ramp2+최대가중치4, 결과 보기 전 원리로 결정).
scripts/ffpp_c23_smoothing_uda_teacher_pool_v2.py, tests/test_smoothing_uda_teacher_pool_v2.py, train_test_ffpp_c23_smoothing_uda_teacher_pool_v2_seed42 MD/7셀 코랩 추가(기존 스크립트 변경 없음).
조건: ce_rank_fixed(대조군), ts_ema(TS 기본), ts_frozen_teacher(교사=baseline 고정), ts_decoupled_head(임시정답은 보조 head만 학습, 교사는 인코더+판정head EMA; DST의 분리 부분만).
주 결과=6epoch final.pt, 보조=best_ffpp.pt(@ffpp). 라벨 없는 검증지표(엔트로피, 교사-학생 일치, SND)와 보조 head adapt-val을 기록만 한다.
CPU 테스트: ce_rank_fixed가 pool v2 CE+Rank, ts_ema가 이전 ts_confidence와 같은 epoch까지 가중치 완전 일치; 고정 교사가 baseline과 동일 유지; 보조 head 손실이 판정 head에 gradient를 주지 않음; 중단 재개 일치; 평가 재사용. GPU 학습 미실행.

## 교사 보호 실험 결과와 도메인/스무딩 AUC 분해 (2026-10-22)
교사 보호(CLIP/DINO, seed42, 고정 6epoch final): 4조건 pooled AUC Baseline 0.796/0.723, CE+Rank고정 0.763/0.703, TS EMA 0.807/0.695, 고정교사 0.795/0.730, head분리 0.818/0.717 (SDA 0.880/0.858).
head분리-TS EMA CLIP +0.010[+0.003,+0.018], DINO +0.022[+0.012,+0.034]로 일관되지만 작고, Baseline 대비 CLIP +0.022/DINO -0.005. 고정교사-EMA는 CLIP -0.012, DINO +0.035로 방향이 갈린다.
교사 임시정답 정확도는 6epoch 내내 고정(CLIP 약75%, DINO 약81%)이고 고정교사와 EMA가 같다: EMA 교사 표류는 없었고 병목은 baseline 교사 품질 자체. CLIP 교사는 가짜 약4,500/진짜 약1,150개를 골라 가짜 쪽 23%·진짜 쪽 32%가 오답, FPR 0.56~0.68.
CE+Rank는 6epoch에서 CLIP 0.814→0.763으로 떨어져(source 과적합) 간격 회복률 기준이 부풀려진다. 라벨 없는 지표(엔트로피/SND/일치도)는 adapt-val과 함께 움직이지 않았다. Xception은 미실행(사용자 판단으로 보류).
scripts/analyze_smoothing_vs_domain_gap.py, tests/test_smoothing_vs_domain_gap.py 추가(CPU, 저장된 cross_condition_auc.csv만 사용, 테스트 라벨 진단 전용·선택 미사용).
분해: matched=같은 조건 진짜·가짜 AUC 평균, 스무딩 몫=matched-pooled16(SDA 대비), 도메인 몫=SDA matched-matched. SDA와의 pooled 차이 중 도메인 몫이 97~100%: CLIP Baseline 0.084=도메인0.081+스무딩0.003, DINO 0.135=0.133+0.002.
스무딩은 판별력을 거의 잃게 하지 않고(원본-원본 대비 강-강 AUC 손실 CLIP 약0, DINO 약0.01) 진짜·가짜 점수를 모두 가짜 쪽으로 민다. 진짜가 더 스무딩된 칸 평균 CLIP 0.737/DINO 0.662 vs 가짜가 더 스무딩된 칸 0.852/0.781이 서로 상쇄되어 pooled AUC는 이 쏠림을 구조적으로 보지 못한다.
쏠림(두 방향 차이)은 CE+Rank가 DINO 0.119→0.007, CLIP 0.115→0.050으로 크게 줄이지만 matched(도메인 판별력)를 잃는다. TS 계열의 CLIP 개선도 도메인 몫 감소(0.081→0.062)다.
해석: 스무딩만 겨냥한 방법(예: 10-15 C 비라벨 타깃 일관성)은 pooled AUC를 올릴 수 없다. 연구 질문 후보 A(스무딩된 진짜 오탐 감소; 주 지표 강진짜vs원본가짜 AUC·고정임계값 FPR) vs B(CDF AUC 향상=일반 도메인 적응). 권고는 A, 사용자 결정 대기.

## 스무딩 점수 이동·임계값 오탐 진단 (2026-10-22)
scripts/analyze_smoothing_threshold_shift.py, tests/test_smoothing_threshold_shift.py, analyze_smoothing_threshold_shift MD/2셀 CPU 코랩 추가. 저장된 CDF test 예측(8,288×4)만 읽는다.
지표: skew(가짜가 더 스무딩된 칸 AUC평균-진짜가 더 스무딩된 칸), 확률0.5 조건별 FPR/FNR, 원본 진짜 FPR5% 임계값에서 강 스무딩 진짜 FPR·짝 뒤집힘, 원본→강 margin 이동(원본 진짜 std 단위). @normal5는 test 진짜 라벨로 맞춘 진단 전용.
결과(skew / 진짜 뒤집힘 / 강진짜 FPR@normal5): Xception Baseline 0.220/17.5%/22.1% → CE+Rank v1 -0.009/1.3%/4.2%(이동 0, 완전 해소);
DINO 0.119/9.6%/14.4% → CE+Rank v1 0.045/3.0%/7.7%, 6ep 0.007/1.4%/5.6%; CLIP 0.115/9.4%/14.4% → CE+Rank v1 0.086/5.5%/10.5%, 6ep 0.050/2.3%/7.1%(진짜 이동 0.39σ 잔존).
CLIP TS EMA/head분리 뒤집힘 3.1%/3.5%. 이전 "CE+Rank는 CLIP에만 효과"는 pooled AUC 기준의 착시이며, 스무딩 지표로는 CLIP에서 효과가 가장 작다.
확률0.5에서 원본 진짜 FPR이 23~45%(CE+Rank Xception 41%, DINO 39%)로 스무딩과 무관한 임계값/도메인 오탐이 크다. 스무딩이 더한 FPR(원본→강)은 Baseline Xception +29.5/CLIP +19/DINO +17%p → CE+Rank -0.5/+16/+7%p.
해석: 스무딩 쏠림은 CE+Rank로 Xception 해소, DINO 대부분, CLIP 절반. 남은 과제는 CLIP 잔여 쏠림과, 스무딩과 별개인 CDF 임계값/도메인 오탐의 분리 서술.
코랩의 pool v2 경로(ce_rank_pooled, sda_new_balanced)가 실제 폴더와 맞지 않아 건너뛰었다. CLIP CE+Rank v1 pooled 0.8135가 pool v2 CE+Rank와 같아 CE+Rank 결론에는 영향 없음. SDA 상한 비교는 경로 수정 후 재실행 필요.

## CDF 저주파 스타일 진단 (FDA, 2026-10-22)
scripts/diagnose_cdf_style_gap.py, tests/test_cdf_style_gap.py, diagnose_cdf_style_gap MD/2셀 GPU 코랩 추가. 학습·선택 없음, 추론만.
CDF test 8,288장(원본·강 스무딩)을 256×256으로 맞추고 채널별 중앙 저주파 Fourier 진폭(β=0.01/0.03)만 교체, 위상·고주파 유지.
변형: resize(대조), ffpp_b(FF++ train 2,000장 평균 진폭), cdf_b(CDF test 2,000장 라벨 없이 평균, 평균 진폭 교체 자체의 대조). 모델: 세 백본 Baseline과 CE+Rank(delta_transfer_v1).
원본 AUC 변화(resize 대비, ffpp_b0.01/cdf_b0.01/ffpp_b0.03/cdf_b0.03): CLIP Baseline -0.063/-0.050/-0.123/-0.090, CE+Rank -0.054/-0.046/-0.106/-0.077;
DINO Baseline -0.042/-0.051/-0.053/-0.032, CE+Rank -0.039/-0.052/-0.070/-0.055; Xception Baseline -0.017/-0.028/-0.014/-0.017, CE+Rank -0.017/-0.025/-0.001/-0.006.
12개 조합 모두 FF++ 스타일로 AUC가 오르지 않았고, ffpp_b와 cdf_b 차이는 ±0.01~0.03로 방향도 일정하지 않다. 변환 시 원본 진짜 FPR이 CLIP 51%→87~98%, DINO 23%→62~85%로 증가(인공 변형을 조작 흔적으로 읽음).
해석: 저주파 스타일은 FF++→CDF 차이의 원인이 아니다. 앞의 분해와 합치면 차이는 조작 흔적(합성 방식·품질) 쪽. FF++→CDF 스타일 짝 학습은 진행하지 않는다.
한계: FDA 연산 손상이 순수 스타일 효과를 가렸을 수 있으나 cdf_b 대비 FF++ 스타일 고유 이득은 어디에도 없었다. 실행 약 2시간(CPU FFT를 6개 체크포인트마다 반복); 이후 공유 변환은 캐시하기로 함.

## Source-only ablation 결과와 held-out 뷰티 필터 보정 (2026-10-23)
scripts/ffpp_c23_smoothing_source_ablation.py(+테스트, 모델별 7셀 코랩): clean_only / aug_single(원본 1/4 또는 무작위 강도 스무딩 한 장) / aug_pair(원본+스무딩 두 장 CE) / aug_pair_rank(두 장 CE + 0.25 순위, pool v2 CE+Rank와 가중치 완전 일치). 같은 baseline·표본·배치, FF++ val pooled AUC 선택(최대10/patience3), CDF는 학습·선택 미사용.
CDF 공식 test(bilateral, seed42) 원본AUC / 스무딩 격차(원본AUC − 강진짜vs원본가짜) / 강진짜vs원본가짜 / 진짜 뒤집힘@normal5:
Xception Baseline .660/.201/.459/17.5%, clean .671/.223/.448/14.8%, single .647/.013/.634/2.0%, pair .653/-.003/.657/1.5%, rank .656/-.011/.667/1.3%.
DINO Baseline .728/.119/.609/9.6%, clean .734/.092/.642/6.9%, single .720/.044/.676/3.3%, pair .725/.023/.701/2.6%, rank(epoch8) .717/.007/.710/1.4%.
CLIP Baseline .798/.116/.682/9.4%, clean .801/.116/.686/9.5%, single .771/.082/.689/5.0%, pair .800/.087/.713/5.5%, rank .813/.084/.729/5.5%.
격차 차이(조건−rank, 영상 부트스트랩95%): Xception single +.024[.022,.026], pair +.008[.007,.009]; DINO single +.037[.032,.042], pair +.016[.013,.020]; CLIP single -.002[-.005,.002], pair +.004[.002,.005].
사전 판정(세 백본 모두 rank 격차 < single·pair, 원본AUC 손실 ≤.01): Xception·DINO 충족, CLIP은 single과 격차 동률이라 미충족이지만 원본AUC·강진짜vs원본가짜는 rank가 최고. clean_only는 격차를 줄이지 못해 효과는 스무딩 노출에서 나온다.
한계: seed42 단일(부트스트랩은 test 표본 변동만), DINO run 변동 ~.03이라 pair vs rank는 seed43/44로 확정 필요(사용자: 나중에). 10/16 matched Aug vs CE+Rank v1에서 짝 증강이 격차가 더 작아 보였던 것은 선택 기준 차이로 보이며 공정 비교에서 재현되지 않았다.
scripts/beauty_smoothing_filters.py, calibrate_beauty_filters.py(+테스트, CPU 코랩): StyleRetoucher(arXiv 2312.14389) 2.1절의 피부 스무딩 필터 세 계열에서 하나씩 — weighted median(Zhang CVPR2014), fast global smoother(Min TIP2014), domain transform(Gastal SIGGRAPH2011), OpenCV contrib 4.11.0.86(학습 OpenCV와 같은 버전).
같은 피부 마스크·섞기 비율(.25/.5/.75). 보정 기준은 FF++ train 300장의 피부 마스크 평균 |RGB 변화| 중앙값을 bilateral 완전 스무딩(2.06/255)에 맞춤. 고주파 에너지 감소율은 c23에서 압축·윤곽 엣지가 지배(bilateral .08)하고 weighted median에서 음수라 기준에서 제외, 참고값만 기록.
보정값: weighted median 반경=ROI 한 변의 1.607%(변화 2.04), fast global 범위=0.511%(2.05, 고주파 감소 .45로 성격이 가장 다름), domain transform 색 시그마 47.1(공간 .02, 2.05). 학습 bilateral 자체가 FF++ 피부를 평균 ~2/255만 바꾸는 육안 거의 불가 수준의 약한 보정임을 기록.
지표 이름 정리(2026-10-23): 스무딩 격차는 양방향으로 보고한다. fp_gap = 원본AUC − AUC(강 스무딩 진짜 vs 원본 가짜)(보정된 진짜, 오탐 방향; 이전 smoothing_gap), fn_gap = 원본AUC − AUC(원본 진짜 vs 강 스무딩 가짜)(보정된 가짜, 미탐 방향). 교차 AUC는 fp_cross_auc/fn_cross_auc, 짝 뒤집힘은 fp_flip/fn_flip@normal5. 불변이려면 두 격차가 모두 0 근처여야 하며, 한쪽만 작으면 쏠림을 반대로 옮긴 것이다. 부트스트랩 파일은 bootstrap_gaps_vs_aug_pair_rank.csv.
source ablation 코드의 평가 부분만 바뀌어도 파일 해시가 학습 config에 들어가 완료된 run이 불일치로 막히던 문제를 고쳤다: 완료된 run은 이 파일 해시만 다를 때 건너뛴다(학습 중인 run 재개는 그대로 엄격).
FP/FN 두 방향 ablation 결과(2026-10-23, 저장 예측 재사용, CDF 공식 test bilateral, seed42) — fp_gap / fn_gap (max|·|):
Xception Baseline .201/-.181(.201), clean .223/-.191(.223), single .013/-.023(.023), pair -.003/-.005(.005), rank -.011/+.005(.011).
DINO Baseline .119/-.091(.119), clean .092/-.070(.092), single .044/-.039(.044), pair .023/-.004(.023), rank .007/+.002(.007).
CLIP Baseline .116/-.089(.116), clean .116/-.086(.116), single .082/-.076(.082), pair .087/-.075(.087), rank .084/-.070(.084).
부트스트랩(조건−rank, fn_gap): Xception pair -.011[-.011,-.010], DINO pair -.006[-.008,-.004], CLIP single -.006[-.009,-.003], pair -.006[-.007,-.004].
해석: rank가 쏠림을 반대로 옮긴 경우는 없다(fn_gap ≤ +.005, fn_flip ≤5.3%). Baseline은 두 방향이 반대 부호(보정이 진짜·가짜 모두를 가짜 쪽으로 민다). DINO는 rank가 두 방향 모두 가장 0에 가깝다. Xception은 pair·rank 모두 사실상 불변이며 rank는 약한 과보정(fp -.011)으로 max 기준 pair가 근소하게 앞선다(.005 vs .011). CLIP은 세 증강 방법 모두 절반 정도만 줄였고(max .082~.087) rank의 장점은 원본AUC(.813)·fp_cross_auc(.729). 결론 표현: 순위 손실은 증강과 같거나 더 불변에 가깝고 판별력은 같거나 더 잘 지킨다(모든 백본에서 증강보다 확실히 낫다는 아님). 판정은 max(|fp_gap|,|fn_gap|)와 두 격차를 함께 보고, seed43/44로 pair vs rank 차이 확인 예정.
Colab 메모: Drive에 저장된 노트북 사본은 git pull로 갱신되지 않아 마지막 셀이 옛 열 이름을 찾는다. 최신 노트북은 Colab에서 GitHub로 다시 연다.

## Held-out 뷰티 필터 일반화 결과와 "덧셈 보정" 해석 (2026-10-23)
evaluate_beauty_filter_generalization(seed42, CDF test, bilateral로만 학습한 ablation 모델) — max(|fp_gap|,|fn_gap|), 필터 순서 bilateral/WMF/FGS/DT:
Xception Baseline .201/.123/.266/.046, single .023/.040/.115/.103, pair .005/.048/.094/.114, rank .011/.051/.085/.122.
DINO Baseline .119/.093/.189/.052, single .044/.029/.117/.013, pair .023/.015/.084/.042, rank .007/.008/.073/.044.
CLIP Baseline .116/.079/.124/.066, single .082/.050/.105/.036, pair .087/.055/.108/.041, rank .084/.054/.102/.039.
DT(원래 Baseline을 거의 흔들지 않는 필터)에서 Xception·DINO 증강 모델은 부호가 뒤집혀 미탐 방향으로 과보정(Xception rank fn_gap +.122, 진짜/가짜 이동 -.44/-.49σ, FN 뒤집힘 10.1% vs Baseline 3.3%; Baseline보다 악화). WMF도 Xception에서 역전. FGS(Baseline 영향이 가장 큰 필터)는 덜 보정. 과보정 크기 Xception>DINO>CLIP, single<pair<rank. 원본 AUC는 불변이고 진짜·가짜가 같은 방향으로 이동.
해석(가설): 단일 필터 증강은 스무딩 불변성이 아니라 "학습 필터의 흔적이 보이면 점수에서 일정량을 빼는" 필터 전용 덧셈 보정을 학습한다. bilateral에서는 정확히 상쇄되어 완벽해 보이지만 처음 보는 필터에서는 과·저보정된다. fp_cross_auc만 보면 DT에서 rank가 최고(.768)로 보여 역전을 놓치므로 판정은 두 방향 격차를 함께 본다.
사용자 결정: 주 실험을 "필터 k로 학습 → 필터 k로 평가"(대각선)로 바꾸고, 같은 모델의 다른 필터 평가(대각선 밖)는 보조로 둔다. 기존 v1 결과는 약한 보정(약 2/255) 참고 결과로 유지한다.

## 스무딩 프로토콜 v2 확정 (2026-10-23)
v1(기하 타원 마스크 + bilateral, 피부 변화 약 2/255)은 육안으로 거의 보이지 않아 뷰티 보정으로 보기 어렵다는 사용자 지적으로 새 프로토콜을 정했다. CDF/FF++ 크롭으로 강도·마스크·필터 설정을 단계별로 육안 검토(outputs/beauty_strength_preview, git 제외)한 뒤 사용자가 확정.
- 마스크: SegFormer jonathandinu/face-parsing(revision 758b82e1, CelebAMask-HQ) 피부+코, 1% 침식·1.2% 페더링. 기하 타원은 강도를 올리면 배경·턱선·머리카락까지 번져 폐기. 무작위 FF++ 50·CDF test 50장 검토에서 피부 면적 CDF 16~34%, FF++ 15~28%; 실패는 얼굴 미검출 fallback 프레임(피부 0%, 원래도 미변환)뿐. 한계: 덥수룩한 수염이 피부로 잡혀 함께 문질러짐(50장 중 약 2장).
- 필터 4개(경계 보존: 색 시그마 고정 bilateral 35 / WMF 35 / FGS 8 / DT 40, 강도는 공간 범위로만): 색 시그마가 크면 코·눈가 음영 경계를 넘어 얼굴이 뭉개지는 것을 확인하고 낮췄다. 육안 승인 공간 범위 bilateral .08225 / WMF .04465 / FGS .0158 / DT .1044(CDF 비테스트 16장·기하 마스크 기준 피부 변화 3.5).
- 약·중·강 = 마스크 안 .25/.5/.75. L3 100%는 뭉개짐, L2(3.5) 75%를 강으로 채택.
- 정식 강도: calibrate_beauty_smoothing_v2.py가 FF++ train 300장·파싱 마스크로 앵커(bilateral 승인값)의 피부 변화 중앙값에 나머지 세 필터의 공간 범위를 맞춘다(CDF 미사용). 결과 Drive manifests/beauty_smoothing_v2/calibration.json(아직 미실행).
- 코드: scripts/beauty_smoothing_v2.py(프로토콜·파서·캐시 리더), build_beauty_smoothing_v2_cache.py(데이터셋 cdf_test/ffpp_test/ffpp_trainval; mask + 필터 4개를 섞기 전 100%로 부분마다 Drive tar 하나, 재개·복원), evaluate_beauty_smoothing_v2.py(cdf_test 또는 ffpp_test, 현재 Baseline만, 원본 1회 + 필터 4×3강도). prepare_ffpp_c23_training_data.py에 --test-only 추가(기존 동작 불변). v1 코드·결과는 변경하지 않음.
- 노트북: calibrate_beauty_smoothing_v2 → build_beauty_smoothing_v2_cache(cdf_test·ffpp_test 먼저, ffpp_trainval은 병렬) → evaluate_beauty_smoothing_v2_baseline_seed42(모델별). tests/test_beauty_smoothing_v2.py(가짜 파서·필터) 통과, 실제 파서·contrib 필터로 CDF 8장 캐시 생성·tar 확인. GPU 실행은 아직 없음.
- 계획: Baseline v2 평가에서 쏠림 확인(관문) → 대각선 학습 aug_single/aug_pair/aug_pair_rank × 필터 4 × 백본 3 = 36 run(seed42), best epoch는 FF++ val 원본+학습 필터 k pooled AUC, 각 모델을 필터 4개 모두로 평가해 4×4 표. clean_only는 주 실험에서 제외(v1에서 추가 학습만으로 격차가 줄지 않음을 이미 보임). FF++ test(같은 도메인) 평가도 Baseline부터 포함: CDF 도메인 차이 없이 스무딩만의 영향을 분리.

## 새 학습법 개발 순서와 D1 진단·CDF 개발 세트 (2026-10-23)
사용자 결정: Baseline v2 평가 뒤 대각선 전에 새 학습법을 먼저 구상·파일럿하고, 괜찮으면 그 방법을 포함해 대각선(필터 4 × 조건 4 × 백본 3)을 만든다.
근거: 점수 손실(CE/pair/rank)은 "스무딩 흔적이 보이면 점수를 깎는" 덧셈 보정으로도 만족되므로, 특징 공간에서 스무딩 흔적을 지우는 방법(특징 일관성, 특징 스무딩 판별기 적대 학습, 여러 필터 혼합)이 후보. CLIP은 v1 대각선에서도 28%만 해결.
- 방법 선택에 CDF 공식 test를 쓰지 않기 위해 cdf_dev 추가: pool v2 membership_labelled.csv(라벨 있는 비테스트, adapt-train+adapt-val), 공식 test 영상과 겹치면 중단. 개발 비교는 cdf_dev + ffpp_test로만 하고 공식 test는 방법 확정 후 대각선에서 한 번. 주의: 이후 adapt-train을 학습에 쓰는 방법은 cdf_dev와 겹치므로 별도 분할 필요.
- 파일럿: bilateral(v2) 행만 aug_single/aug_pair/aug_pair_rank/새 방법 × CLIP·Xception, 대각선과 같은 설정으로 두어 run을 재사용.
- D1(scripts/diagnose_smoothing_feature_probe.py, 노트북 diagnose_smoothing_feature_probe): v1 ablation 체크포인트, FF++ val 1,200장, 원본 vs v1 스무딩 강(bilateral + held-out 3개)의 분류 직전 특징으로 영상 단위 5겹 선형 프로브(스무딩 감지 AUC), bilateral 프로브의 다른 필터 전이 AUC, 특징 이동량, 점수 이동. 예측: 보정이면 증강 모델의 감지 AUC가 높게 유지되고 DT에서 bilateral 프로브가 켜지며 점수가 진짜 쪽으로 이동. 테스트 미사용, GPU 실행 전.
- 연구 목표 정리(사용자 질문): "강건한 탐지"의 성능 지표는 스무딩된 상황의 판별력(교차 AUC, 특히 두 방향 중 나쁜 쪽)이고, 원본 CDF AUC 향상은 도메인 적응 문제라 주 목표가 아니다(원본 AUC는 유지 조건).

## D1 특징 프로브 결과: 불변성 없음, Xception은 필터와 무관한 거의 일정한 점수 차감 (2026-10-23)
FF++ val 1,200장, v1 ablation seed42, 원본 vs v1 강(.75). Drive evaluations/smoothing_feature_probe_v1/<모델>/feature_probe.csv.
- 감지 AUC(bilateral / DT): Xception Baseline .940/.915 → rank .913/.899; DINO .791/.751 → .760/.730; CLIP .762/.731 → .761/.741. 증강 후에도 거의 그대로(최대 -.03) → 세 백본 모두 "스무딩에 둔감해짐(불변성)"은 아님.
- bilateral 프로브의 다른 필터 전이 AUC: Xception .86~.88, DINO .71~.75, CLIP .70~.73 → 필터들이 공통 흔적 방향을 공유.
- 특징 이동량(bilateral): Xception .554→.203, DINO .332→.213, CLIP .364→.271 (줄지만 감지 정보는 남음).
- 점수 이동(raw margin, 원본 진짜 std 약 4~6). Xception Baseline→rank 차감량(진짜/가짜): bilateral 4.0/5.7, WMF 3.3/5.2, DT 3.4/5.8, FGS 2.4/3.7. 필터마다 원래 이동량이 달라도 차감량이 거의 일정 → DT 가짜 +0.98 → -4.86(약 0.85σ 진짜 쪽, CDF DT 미탐 증가와 일치), bilateral에서도 가짜 -1.31. aug_single/pair도 같은 양상(DT 가짜 -3.1/-4.6).
- DINO는 같은 양상이 약함(가짜 차감 bilateral 1.11 ≈ DT 1.13 → DT 가짜 -0.62; FGS 진짜 +1.57 남음 = 덜 보정).
- CLIP은 진짜 쪽 이동만 절반 정도 줄고(bilateral +4.00→+2.17, DT +2.58→+1.26) 가짜 쪽 차감·역전 없음 → 보정도 불변성도 약함(증강에 거의 반응하지 않음). CDF 결과(절반 해소, 역전 없음)와 일치.
해석: 덧셈 보정 가설을 Xception에서 정량적으로 지지(감지 정보 유지 + 필터와 무관한 일정 차감), DINO 약하게 지지, CLIP은 별도 문제(증강 학습이 특징 반응을 거의 바꾸지 못함). 한계: 선형 프로브, FF++ val, seed42, raw margin 척도가 모델마다 다름.
방법 함의: 점수 손실이 아니라 특징 단계에서 흔적을 제거해야 함. 전이 AUC가 높아 공유 방향 제거(후보 3: 사후 투영)로 빠른 인과 확인 가능, 본 방법 후보 1(특징 일관성+rank)/후보 2(특징 판별기 GRL). CLIP은 학습 블록 수(마지막 2개) 한계 가능성 별도 확인.

## 후보 3: 스무딩 방향 제거 진단 구현 (2026-10-23)
scripts/diagnose_smoothing_direction_removal.py(+테스트, 노트북 diagnose_smoothing_direction_removal). v1 ablation seed42 Baseline/aug_pair/aug_pair_rank × 3백본. FF++ train 1,200장의 (bilateral 강 − 원본) 특징 차이 상위 k(0/1/4/16) SVD 방향을 투영으로 제거, 로지스틱 분류기만 FF++ train 원본으로 재학습(백본 고정; k=0 = 재학습 대조군). FF++ val 1,200장 원본 vs v1 강 4필터로 원본 AUC, fp/fn gap, 점수 이동(σ), 제거 후 감지 AUC. 처음 보는 필터는 방향 추정에 쓰지 않음, CDF·test 미사용. 목적: D1의 상관(흔적 유지 + 일정 차감)을 인과로 확인 — 흔적 방향을 지우면 DT 역전·잔여 쏠림이 줄어드는가. 세 백본을 한 실행에서 돌려 뷰를 한 번만 계산(D1에서 모델마다 뷰를 다시 만든 비효율 수정). GPU 실행 전.

## v2 캐시 속도와 weighted median 비결정성 (2026-10-23)
- FF++ 크롭이 커서(한 변 중앙값 299, 상위 10% 587, 1% 945px; CDF 평균 266) 공간 범위가 크기에 비례하는 bilateral 비용이 크기^4로 증가(Mac 1스레드 256px 0.09s, 600px 2.7s, 800px 8.5s). ffpp_trainval 62,418장은 T4(2코어)에서 필터만 하루 이상 → 코어가 많은 런타임 권장. 캐시 워커마다 OpenCV 스레드 1개로 고정(중첩 스레드 과다 방지; bilateral 결과 동일 확인).
- OpenCV contrib weightedMedianFilter는 같은 입력·같은 스레드 수에서도 실행마다 출력이 다름(픽셀 약 55%가 다르나 5단계 초과는 약 4%, 피부 변화량 3.25±0.01). 따라서 WMF 결과는 비트 단위 재현이 안 되며, 한 번 만든 캐시 tar가 기준이다(재계산 금지). 보정값에는 영향 미미.

## 후보 3 결과: 흔적 방향 제거로 보정과 잔여 쏠림이 사라짐 (2026-10-24)
FF++ val(동일 도메인, 원본 AUC .94~.99로 천장 근처라 AUC 격차보다 점수 이동 σ가 유용). 점수 이동 = 강 스무딩 − 원본, 원본 진짜 std 단위(진짜/가짜). model_head = 원래 모델, refit = 방향 제거 없이 분류기만 FF++ train 1,200장 원본으로 재학습(대조군), kN = bilateral 방향 N개 제거 + 재학습.
- Xception rank DT 가짜(역전): model -0.85σ → refit -0.30 → k1 -0.02. 보정의 상당 부분이 분류기 가중치에 있고(재학습만으로 -0.85→-0.30), bilateral 방향 1개 제거로 역전이 사라짐 → D1의 "흔적 감지 후 차감"을 인과적으로 지지.
- Baseline(증강 없음)에 방향만 제거해도 bilateral 쏠림 대부분 소거: Xception 진짜 1.14→0.12σ(k4), DINO 0.61→0.13(k4), CLIP 0.67→0.13(k4). WMF도 크게 감소. FGS는 일부만(Xception 1.57→0.85, DINO 0.97→0.47, CLIP 0.74→0.29; k4) → FGS는 bilateral과 다른 방향 성분이 있음. DT는 Baseline 제거 시 소폭 과보정(Xception -0.29/-0.33σ, DINO -0.15σ; rank의 -0.85보다 작음).
- CLIP rank: refit .54/.11 → k1 .03/-.03(bilateral), FGS .66→.24, DT .32/.06→-.07/-.13, 원본 AUC는 refit과 동일(.899). CLIP의 남은 쏠림은 학습 범위 한계라기보다 점수 손실이 흔적 방향을 지우지 못한 탓일 가능성.
- 비용: 원본 AUC 하락. 상당 부분은 1,200장 로지스틱 재학습 자체(refit: CLIP .942→.932/.940→.899, DINO .955→.941/.953→.939). 재학습 대비 추가 손실은 Xception -.007~-.012, CLIP ~0, DINO k4는 오히려 +.007.
- 제거 후 감지 AUC는 줄지만 0.5까지는 아님(Xception rank .913→.724 k4, .606 k16) → 흔적이 다차원·일부 비선형.
함의: 특징 단계 흔적 제거가 보정을 끊고 처음 보는 필터에도 부분적으로 일반화. 학습법 A는 이를 학습 중에 전체 데이터로 하는 형태(후보 1 특징 일관성, 또는 방향 제거를 학습에 통합)로, FGS처럼 다른 방향을 가진 필터는 남는 과제. 한계: FF++ val, 작은 재학습 표본, seed42, v1 스무딩.

## v2 Baseline 결과와 대각선 학습 코드·λ 파일럿 (2026-10-24)
v2 Baseline CDF test max|gap|(방향) bilateral/WMF/FGS/DT: CLIP .29/.31/.31/.23(모두 오탐 방향, FP 뒤집힘 23~33%), Xception .24/.10/.37/.09(DT만 미탐 방향: FN 뒤집힘 11%, FGS FP 뒤집힘 45%), DINO .06/.07/.27/.14(DT 미탐 방향 FN 11.6%). CLIP FF++ test(같은 도메인) FP gap .13~.22, FP 뒤집힘 19~33% → 도메인 차이와 무관한 스무딩 쏠림. 방향은 필터·백본에 따라 다름(DT는 합성 흔적 세탁 가설). Xception/DINO FF++ test 기록은 결과 확인 후.
v2 캐시 20개 tar 완료(ffpp_trainval 62,418장 34.8GB, ffpp_test 11,058, cdf_test 7,999, cdf_dev).
학습 코드 scripts/ffpp_c23_smoothing_v2_diagonal.py: 필터 k의 v2 캐시(mask+k만 풀기)로 source ablation과 같은 표본·배치·flip·강도 추첨·선택 규칙(FF++ val 원본+k 약중강 pooled AUC). 조건 aug_single/aug_pair/aug_pair_rank/aug_pair_rank_fc(학습법 A: rank + λ·(1−cos(f_smooth, stopgrad f_clean)), 분류기 입력 특징, 검출 프레임만). λ마다 run 이름 aug_pair_rank_fc<λ>. 모든 seed가 baseline seed42에서 출발(--baseline-seed), 에폭 표본 CSV는 seed별 폴더. 평가는 evaluate_beauty_smoothing_v2.py --runs-root로 strategy@train_filter 라벨, held_out 표시. 노트북 train_eval_ffpp_c23_smoothing_v2_diagonal(파일럿·본 실험 공용, 끝난 run 재사용).
파일럿(사용자 결정): bilateral, CLIP·Xception, seed42, aug_pair_rank + rank+일관성 λ∈{1,10,100}, cdf_dev만 평가. A가 rank 대비 처음 보는 필터 쏠림을 줄이고 원본 AUC를 유지하는 λ를 고른 뒤 본 실험(필터 4 × 조건 4 × 백본 3 × seed 3)에 그 λ만 사용.
v2 Baseline FF++ test(같은 도메인) 보충: Xception max|gap| bil .069/WMF .030/FGS .155/DT .015(원본 AUC .980; FP 뒤집힘 25.6/12.6/48.6/7.3%, DT 가짜 이동 -0.65σ·FN 뒤집힘 6.7%), DINO .024/.028/.154/.032(.952; FGS FP 뒤집힘 32.5%, DT 가짜 -0.33σ·FN 뒤집힘 9.2%). CDF와 같은 패턴: FGS가 세 백본 모두 최대 오탐 방향, DT는 Xception·DINO에서 두 도메인 모두 진짜 쪽(세탁) → 도메인 효과가 아님. FF++ test는 원본 AUC 천장(.95~.98)이라 AUC 격차는 작게, 뒤집힘·점수 이동은 크게 나타남.

## λ 파일럿 CLIP 결과 (bilateral 학습, seed42, cdf_dev; 2026-10-24)
원본 AUC: Baseline .796, rank .775, fc1 .766, fc10 .792, fc100 .761. 나쁜 쪽 교차 AUC(bil/WMF/FGS/DT): rank .706/.681/.612/.744, fc1 .740/.710/.616/.735, fc10 .743/.706/.630/.761, fc100 .754/.709/.621/.739 (Baseline .505/.487/.470/.574).
max|gap|: rank .069/.095/.163/.032, fc1 .027/.057/.150/.031, fc10 .049/.086/.162/.030, fc100 .007/.052/.140/.022.
강 스무딩 점수 이동 σ(진짜/가짜) bilateral: rank .36/.19, fc1 .12/.08, fc10 .24/.18, fc100 .03/-.01; 강도 곡선 fc100은 약·중·강 모두 0 근처(평탄 = 불변성 형태). CLIP rank는 약한 강도 과보정 없음(미보정 형태, v1과 같음).
흔적 프로브(FF++ val, bilateral 강) 감지 AUC/특징 이동: Baseline .921/.687, rank .927/.330, fc1 .884/.172, fc10 .929/.228, fc100 .769/.151.
해석: 일관성 항이 학습 필터 쏠림과 흔적을 줄이고(λ=100에서 뚜렷) 교차 AUC를 올림. 처음 보는 필터 WMF는 절반 감소, FGS는 거의 그대로(흔적 방향이 다름 — 후보 3과 일치), DT는 원래 작음(fc1/fc100 가짜 -0.10~-0.12σ 소폭). λ 순서가 단조롭지 않음(fc10이 fc1보다 쏠림·감지 AUC가 큼) → seed 하나의 run 변동이 큼. λ 결정은 Xception 결과와 함께.

## λ 파일럿 Xception 결과: λ=100은 학습이 무너져 Baseline이 선택됨, λ=1/10은 처음 보는 필터 과보정을 못 줄임 (2026-10-24)
cdf_dev, bilateral 학습, seed42. 원본 AUC: Baseline .787, rank .744, fc1 .729, fc10 .717, fc100 = Baseline(.787).
- fc100: epoch1에 CE 1.04·순위 .60으로 급등(일관성 .0063×100), FF++ val pooled AUC .751→.872→.903으로 회복 중이었으나 epoch0(Baseline)을 넘지 못해 patience 3으로 종료 → best.pt = epoch0. 즉 λ=100은 Xception 전체 미세조정에서 초기 충격이 커서, 조기종료 규칙 아래 학습 결과가 남지 않음.
- 학습 필터 bilateral max|gap|/나쁜 쪽 교차 AUC: rank .019/.750, fc1 .029/.734, fc10 .021/.726 (Baseline .236/.551).
- 처음 보는 필터 max|gap|(방향): WMF rank .152(미탐: 가짜 -0.69σ) fc1 .184 fc10 .190; FGS rank .102 fc1 .114 fc10 .095; DT rank .353(가짜 -1.36σ, Baseline도 -0.37σ 세탁 방향) fc1 .390 fc10 .387. → 일관성 항이 Xception의 과보정을 줄이지 못함.
- 흔적 프로브(FF++ val) 감지 AUC: Baseline .975, rank .941, fc1 .809, fc10 .760 → 흔적은 줄었지만 0.5와 거리가 멀고, 줄어든 만큼 과보정이 줄지 않음.
해석: 코사인 일관성(1-cos)은 큰 성분이 지배하는 각도만 묶어서(값이 0.006 수준), 분류기가 증폭할 수 있는 작은 크기의 흔적 방향은 남김 → 선형 프로브로 여전히 구분 가능. CLIP(마지막 2블록만 학습)에서는 통했지만 Xception(전체 학습)에서는 불충분. 다음 후보: λ 워밍업(초기 충격 방지)으로 λ=100 재시도, 또는 흔적을 직접 겨냥하는 후보 2(특징 스무딩 판별기 적대 학습)/표준화 특징 공간의 일관성.
λ 워밍업 추가(사용자 결정): --fc-ramp-epochs R이면 처음 R epoch 동안 λ를 스텝마다 0→λ로 선형 증가, run 이름 aug_pair_rank_fc<λ>_ramp<R>, config에 fc_ramp_epochs(워밍업 없는 기존 run의 config는 그대로라 재사용 유지). Xception만 λ=100·R=2로 빠르게 재확인(CLIP 재실행 안 함).
Xception λ=100 워밍업 2 결과: epoch1 FF++ val pooled .966(epoch0)→.724, epoch2 .916, epoch3 .935로 회복 중 patience 3에 걸려 다시 epoch0 선택. 사용자 결정으로 워밍업 epoch은 patience에 세지 않도록 변경(config patience_rule), 같은 이름 run을 --replace-runs로 재학습(기존 폴더는 *_replaced_<시각>으로 보존). 평가는 체크포인트가 바뀐 예측을 재계산.
