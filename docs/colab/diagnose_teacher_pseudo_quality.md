# Teacher_Pseudo_Quality (GPU, 추론만, 학습 없음)

**질문:** TS의 교사를 Baseline 대신 **CE+Rank**로 두면 임시 정답이 깨끗해지는가? 특히 "스무딩된 진짜 → 가짜" 오답이 줄어드는가?
B(라벨 없는 스무딩 도메인 적응)를 학습하기 전에 이걸 먼저 확인한다.

- 대상: pool v2 **adapt-train** 전체 프레임(UDA 때 라벨 없이 쓰는 이미지) × 원본/약/중/강 스무딩.
- 교사: 세 백본 각각 Baseline, CE+Rank(`delta_transfer_v1`, FF++ val로 고른 모델).
- 규칙: TS와 같음. 가짜 확률이 0.9 이상이면 "가짜", 0.1 이하면 "진짜" 임시 정답(0.95도 함께 기록).
- 라벨은 **임시 정답 채점에만** 사용. 학습·모델 선택 없음. 공식 test 이미지는 쓰지 않는다.

|지표|뜻|
|---|---|
|`selected_fraction`|확신 기준을 넘어 임시 정답이 붙은 비율|
|`pseudo_accuracy`|붙은 임시 정답의 정확도|
|`selected_real` / `selected_fake`|"진짜" / "가짜"로 붙은 수 (쏠림 확인)|
|`fake_side_error`|"가짜" 임시 정답 중 실제 진짜 비율|
|`true_real_called_fake`|실제 진짜 중 확신 있게 "가짜"로 붙은 비율 (**조건별로 보면 스무딩 편향**)|
|`true_fake_called_real`|실제 가짜 중 확신 있게 "진짜"로 붙은 비율|

- bilateral 결과는 이미지당 한 번만 계산해 `/content` 캐시에 두고 6개 교사·3개 강도가 함께 쓴다.
- 예측은 Drive `evaluations/teacher_pseudo_quality_v1/`에 저장. 끊기면 다시 실행하면 끝난 예측은 건너뛴다.
- 진행 막대가 셀에 보이도록 `!python -u`로 실행한다.

## Cell 1 — 코드·CDF non-test 준비

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
assert (PROJECT/'scripts/diagnose_teacher_pseudo_quality.py').is_file(),'새 코드를 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)

ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
TARGET_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
archives={'celebdf_v2_roi16_non_test_real_v1.tar':826142720,'celebdf_v2_roi16_non_test_fake_v1.tar':5679564800}
for name,size in archives.items():
    manifest=TARGET_ROOT/'manifests'/f'{name.removesuffix(".tar")}.csv'
    if manifest.is_file():
        print('Already extracted:',name,flush=True); continue
    source=ROOT/'data/celebdf_v2_face_crops16_v1'/name
    assert source.is_file() and source.stat().st_size==size,f'TAR 확인: {source}'
    assert shutil.disk_usage('/content').free>2*size+2*2**30,'Colab 로컬 공간 부족'
    local=Path('/content')/name
    print('Copying:',name,flush=True); shutil.copy2(source,local)
    assert local.stat().st_size==size,'복사 미완료'
    print('Extracting:',name,flush=True)
    subprocess.run(['tar','-xf',str(local),'-C','/content'],check=True); local.unlink()
    assert manifest.is_file(),manifest

candidates=[ROOT/'manifests/ffpp_c23_smoothing_rank_sda_pool_v2/seed42'/m/'pool_v2/membership_labelled.csv' for m in ['clip_b16','dinov2_b14','xception']]
MEMBERSHIP=next(p for p in candidates if p.is_file())
print('pool v2 membership:',MEMBERSHIP,flush=True)
```

## Cell 2 — 교사 품질 진단

```python
OUTPUT=ROOT/'evaluations/teacher_pseudo_quality_v1'
!python -u "{PROJECT}/scripts/diagnose_teacher_pseudo_quality.py" --membership "{MEMBERSHIP}" --target-root "{TARGET_ROOT}" --baseline-runs-root "{ROOT}/runs/ffpp_c23_4methods_baseline_v1" --ce-rank-runs-root "{ROOT}/runs/ffpp_c23_smoothing_delta_transfer_v1" --output-root "{OUTPUT}" --models clip_b16 dinov2_b14 xception --thresholds 0.9 0.95 --batch-size 64 --workers 2
assert _exit_code==0,'진단 실패: 위 오류 확인 후 다시 실행하면 끝난 예측은 건너뜀'
```
