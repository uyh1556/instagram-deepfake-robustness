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
