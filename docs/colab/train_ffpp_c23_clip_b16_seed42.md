# FFPP_c23_CLIP_ViTB16_Baseline_Seed42

1 notebook = 1 model. 새 FF++ c23 native ROI16 데이터로 source baseline을 학습하고 FF++ Test까지 평가한다.
Celeb-DF·스무딩·augmentation·UDA는 이 노트북에 포함하지 않는다.

## 학습 설정

- Real·Deepfakes·Face2Face·FaceSwap·NeuralTextures, 같은 고정 source-aware 분할. 공식 FF++ split은 아님.
- 매 epoch Real:전체 Fake =1:1, Fake 네 기법 균등. 영상별로 고르게 선택하고 epoch마다 바꾼다.
- 검증·테스트는 저장된 분할 이미지 전체로 고정. 이미지 단위 AUC/FPR/FNR/F1/Accuracy와 기법별 AUC를 저장.
- 최초1epoch 분류기만 학습. 이후 마지막2개 Transformer 블록 + 최종 LayerNorm·visual projection + 분류기 fine-tuning. 나머지 인코더 동결.
- 인코더 LR 1e-5, 분류기 LR1e-4. AdamW, 최대15epochs, patience4, seed42, batch16.
- 최고 모델: **FF++ clean Validation의 네 기법 평균 frame ROC-AUC**. Test는 모델 선택에 사용하지 않는다.
- 좌우반전0.5만 학습에 적용. JPEG/스무딩/노이즈를 추가하지 않는다.
- Native crop에서 OpenAI CLIP bicubic224/center crop + CLIP mean/std 정규화. 512차원 visual feature에 Linear512→2, text encoder/추가 L2 normalize 없음. 얼굴 검출실패 fallback은 그대로 포함한다.
- BF16 지원 GPU에서는 BF16, 그 외 FP32. FP16은 사용하지 않는다.

## 준비

새 코드를 직접 push하고, 아래 폴더에 **기법별 Train/Val TAR 5개 + 기존 Test TAR + method_inventory.json**을 업로드한다.
`MyDrive/instagram-deepfake-robustness/data/ffpp_c23_4methods_roi16_v1/`

- ffpp_c23_real_roi16_trainval_v1.tar
- ffpp_c23_deepfakes_roi16_trainval_v1.tar
- ffpp_c23_face2face_roi16_trainval_v1.tar
- ffpp_c23_faceswap_roi16_trainval_v1.tar
- ffpp_c23_neuraltextures_roi16_trainval_v1.tar
- ffpp_c23_4methods_roi16_test_v1.tar
- method_inventory.json

METHOD_SHA256SUMS.txt·protocol.json도 같은 Drive 폴더에 보관하면 된다.
Test TAR는 기존 파일 그대로 재사용한다. 기존 통합 Train/Val TAR는 새 코랩에 필요하지 않다.
Canonical manifest 바이트·이미지·분할은 그대로 유지하므로 학습 코드/조건과 체크포인트 호환성은 바뀌지 않는다.
기존 c0나 이전 aligned256 TAR를 사용하지 않는다. 모델 초기 가중치는 첫 실행 시 다운로드한다.
GPU는 A100을 권장하며, 부족하면 batch-size를 변경하기 전에 새 run 경로를 사용한다.

## Cell 1 — Drive·코드·설치

kernel에서는 PIL/torch/transformers를 import하지 않는다. 설치와 학습은 별도 Python 프로세스여서 Pillow 재시작 검사를 넣지 않는다.

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess,sys,json
PROJECT=Path('/content/instagram_deepfake_robustness')
if PROJECT.exists():
    subprocess.run(['git','-C',str(PROJECT),'pull','--ff-only'],check=True)
else:
    subprocess.run(['git','clone','https://github.com/uyh1556/instagram-deepfake-robustness.git',str(PROJECT)],check=True)
assert (PROJECT/'scripts/train_ffpp_c23_baseline.py').is_file(),'새 코드 push 필요'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-training.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
```

## Cell 2 — 변수

```python
MODEL='clip_b16'
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
DRIVE_DATA=ROOT/'data/ffpp_c23_4methods_roi16_v1'
DATA_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
RUNS_ROOT=ROOT/'runs/ffpp_c23_4methods_baseline_v1'
RUN=RUNS_ROOT/f'{MODEL}_ffpp_c23_roi16_baseline_v1_seed42'
COMMON=f'--model {MODEL} --data-root "{DATA_ROOT}" --output-root "{RUNS_ROOT}" --seed 42 --epochs 15 --patience 4 --warmup-epochs 1 --train-blocks 2 --batch-size 16 --workers 2 --encoder-lr 1e-5 --head-lr 1e-4'
print('Model:',MODEL,'Run:',RUN)
```

## Cell 3 — 기법별 TAR 준비

Drive TAR를 로컬로 복사한 뒤 추출하며, 이미 추출된 데이터는 재사용한다.
데이터 파일 전체 내용 해시를 반복 계산하지 않고, 학습 시작 시 목록·개수·분할·파일 존재 여부만 확인한다.

```python
!python -u "{PROJECT}/scripts/prepare_ffpp_c23_training_data.py" --drive-data "{DRIVE_DATA}" --data-root "{DATA_ROOT}"
assert _exit_code==0,'데이터 준비 실패: 위 오류 확인'
```

## Cell 4 — 학습

Train/Validation에 batch 진행률·경과시간·ETA가 표시된다. epoch당 요약 한 줄만 추가 출력한다.
중단 시 같은 셀 재실행: last.pt의 완료 epoch부터 재개. 완료 run은 건너뛴다.
실험 설정이 달라지면 기존 결과를 덮어쓰지 않고 중단한다.

```python
!python -u "{PROJECT}/scripts/train_ffpp_c23_baseline.py" train {COMMON}
assert _exit_code==0,'학습 실패: 위 오류 확인'
```

## Cell 5 — FF++ Test

best.pt를 사용한다. 인물/source 연결이 분리된 FF++ test이며 Celeb-DF test는 아니다.

```python
!python -u "{PROJECT}/scripts/train_ffpp_c23_baseline.py" test {COMMON}
assert _exit_code==0,'FF++ 평가 실패: 위 오류 확인'
```

## Cell 6 — 결과 확인

```python
result=json.loads((RUN/'test_metrics.json').read_text())
print('FF++ macro4 frame AUC:',round(result['macro_method_auc'],4))
print('Real FPR:',f"{result['frame']['real_fpr']:.2%}", 'Fake FNR:',f"{result['frame']['fake_fnr']:.2%}")
print('Accuracy:',f"{result['frame']['accuracy']:.2%}", 'F1:',round(result['frame']['f1'],4))
for method,metrics in result['per_method'].items():
    print(method,'AUC=',round(metrics['auc'],4),'FNR=',f"{metrics['fake_fnr']:.2%}")
print('Saved:',RUN)
```
