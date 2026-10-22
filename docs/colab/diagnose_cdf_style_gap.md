# CDF_Style_Gap (GPU, 추론만, 학습 없음)

**질문:** FF++→CDF 성능 차이 중 "색감·밝기 같은 저주파 스타일" 몫이 얼마나 되는가?
CDF test 이미지의 저주파 Fourier 진폭만 FF++ 평균 진폭으로 바꿔(FDA) 이미 학습된 모델에 넣어 본다.
위상과 고주파(조작 흔적)는 그대로 둔다.

|변형|내용|
|---|---|
|`resize`|256×256으로 바꾸기만 함 (대조군)|
|`ffpp_b0.01`, `ffpp_b0.03`|저주파 진폭을 **FF++ train 평균**으로 교체 (β = 교체 영역 크기)|
|`cdf_b0.01`, `cdf_b0.03`|같은 연산을 **CDF 자기 평균**으로 (평균 진폭으로 바꾸는 것 자체의 효과 대조)|

- 모델: Xception / CLIP / DINO 각각 Baseline과 CE+Rank(`delta_transfer_v1`의 FF++ val 선택 모델). 학습·모델 선택 없음.
- 조건: 원본, 강 스무딩. 지표: 원본 AUC, 강 AUC, 강 진짜 vs 원본 가짜 AUC, 확률 0.5 FPR.
- **읽는 법:** `ffpp_b`의 원본 AUC가 `resize`·`cdf_b`보다 **크게** 오르면(예: SDA와의 차이 CLIP ~0.08, DINO ~0.13 중 상당 부분) 스타일이 도메인 차이의 큰 몫 → 타깃 스타일 짝 학습을 해볼 가치가 있다.
  거의 그대로면 차이는 조작 흔적 쪽 → 이 방향은 접는다.
- CDF test 라벨은 이 진단 점수 계산에만 쓴다. `cdf_b` 기준 진폭도 test 이미지(라벨 없이)로 만든다.
- 런타임: **GPU**. 결과: `evaluations/cdf_style_gap_diagnostic_v1/summary.csv`. 중간에 끊겨도 다시 실행하면 끝난 예측은 건너뛴다.

## Cell 1 — 코드·데이터 준비

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess, sys, shutil

PROJECT=Path('/content/instagram_deepfake_robustness')
if PROJECT.exists():
    subprocess.run(['git','-C',str(PROJECT),'pull','--ff-only'],check=True)
else:
    subprocess.run(['git','clone','https://github.com/uyh1556/instagram-deepfake-robustness.git',str(PROJECT)],check=True)
assert (PROJECT/'scripts/diagnose_cdf_style_gap.py').is_file(),'새 코드를 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)

ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
SOURCE_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
TARGET_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
subprocess.run([sys.executable,'-u',str(PROJECT/'scripts/prepare_ffpp_c23_training_data.py'),'--drive-data',str(ROOT/'data/ffpp_c23_4methods_roi16_v1'),
                '--data-root',str(SOURCE_ROOT),'--trainval-only'],check=True)
name,size='celebdf_v2_roi16_test_v1.tar',557015040
if not (TARGET_ROOT/'manifests/celebdf_v2_roi16_test_v1.csv').is_file():
    source=ROOT/'data/celebdf_v2_face_crops16_v1'/name
    assert source.is_file() and source.stat().st_size==size,f'TAR 확인: {source}'
    local=Path('/content')/name
    print('Copying:',name,flush=True); shutil.copy2(source,local)
    assert local.stat().st_size==size,'복사 미완료'
    subprocess.run(['tar','-xf',str(local),'-C','/content'],check=True); local.unlink()
print('Data ready',flush=True)
```

## Cell 2 — 진단 실행

```python
OUTPUT=ROOT/'evaluations/cdf_style_gap_diagnostic_v1'
subprocess.run([sys.executable,'-u',str(PROJECT/'scripts/diagnose_cdf_style_gap.py'),
    '--source-root',str(SOURCE_ROOT),'--target-root',str(TARGET_ROOT),
    '--baseline-runs-root',str(ROOT/'runs/ffpp_c23_4methods_baseline_v1'),
    '--ce-rank-runs-root',str(ROOT/'runs/ffpp_c23_smoothing_delta_transfer_v1'),
    '--output-root',str(OUTPUT),'--models','clip_b16','dinov2_b14','xception',
    '--betas','0.01','0.03','--size','256','--reference-images','2000','--batch-size','64','--workers','2'],check=True)
print('Saved:',OUTPUT/'summary.csv')
```
