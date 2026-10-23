# Source_Ablation_Seed42 (모델 공용: Xception / CLIP / DINO)

**질문:** 순위 손실(`aug_pair_rank`)이 단순 스무딩 증강보다 스무딩 시프트를 더 줄이는가?
Cell 2의 `MODELS`에 모델 **하나**를 넣는다. 모델마다 이 노트북을 따로 열어 **동시에** 실행해도 된다.

|조건|FF++ 추가 학습 방식|
|---|---|
|`clean_only`|원본만 (추가 학습 자체의 효과, 대조군)|
|`aug_single`|이미지마다 원본(1/4) 또는 무작위 강도 스무딩 **한 장**|
|`aug_pair`|원본 + 스무딩 **두 장 모두** (순위 손실 없음)|
|`aug_pair_rank`|원본 + 스무딩 두 장 + 순위 손실 0.25 (= 기존 pool v2 CE+Rank와 완전히 같은 설정)|

- 네 조건 모두 같은 FF++ baseline에서 독립 시작하고, epoch 표본·배치 순서·학습 범위·optimizer가 같다. **손실만 다르다.**
- 선택: FF++ val 1,200장 × 4조건 pooled AUC 최고 epoch, 최대 10 epoch, patience 3. **CDF 이미지는 학습·선택에 쓰지 않는다.**
- 평가: CDF **공식 test** 8,288장 × 원본/약/중/강.
  - **주 지표 = 양방향 스무딩 격차** (도메인 차이는 양쪽에 같이 들어가 빠진다)
    - 진짜 쪽(오탐 방향, `smoothing_gap`) = 원본 AUC − (강 스무딩 진짜 vs 원본 가짜) AUC
    - 가짜 쪽(미탐 방향, `fake_side_gap`) = 원본 AUC − (원본 진짜 vs 강 스무딩 가짜) AUC
    - 불변이려면 **둘 다 0 근처**여야 한다. 한쪽만 작으면 쏠림을 반대로 옮긴 것이다.
  - 보조: 쏠림(skew), 진짜·가짜 뒤집힘률, 진짜·가짜 점수 이동. 판별력 확인용으로 원본 AUC를 함께 본다.
  - Cell 6을 다시 실행하면 저장된 예측을 재사용해 가짜 쪽 지표까지 다시 계산한다(추론 없음).
  - 영상 단위 부트스트랩: 각 조건의 스무딩 격차 − `aug_pair_rank`의 스무딩 격차.
- **사전에 정한 판정:** 세 백본 모두에서 `aug_pair_rank`의 스무딩 격차가 `aug_single`·`aug_pair`보다 작고, 원본 AUC가 0.01 넘게 떨어지지 않으면 순위 손실의 기여가 있다고 본다.

**동시 실행·중단 대비**
- 결과 폴더의 첫 새 단계가 모델별이다: `runs|manifests|evaluations/ffpp_c23_smoothing_source_ablation_<모델>/`. Cell 2가 Drive 중복 폴더를 검사한다.
- epoch마다 Drive에 저장. 끊기면 Cell 1부터 다시 실행하면 이어 학습하고 완료 조건은 건너뛴다(`START NEW` / `RESUME` / `ALREADY COMPLETED` 표시).
- CDF 스무딩 이미지는 `/content` 캐시에 한 번만 만들고 다섯 checkpoint가 함께 쓴다. 예측은 Drive에 저장되어 재실행 시 재사용한다.

## Cell 1 — 설치·코드 갱신

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess, sys, json, shutil

PROJECT=Path('/content/instagram_deepfake_robustness')
if PROJECT.exists():
    subprocess.run(['git','-C',str(PROJECT),'pull','--ff-only'],check=True)
else:
    subprocess.run(['git','clone','https://github.com/uyh1556/instagram-deepfake-robustness.git',str(PROJECT)],check=True)
assert (PROJECT/'scripts/ffpp_c23_smoothing_source_ablation.py').is_file(),'새 코드를 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
```

## Cell 2 — 모델·경로·중복 폴더 검사

```python
MODELS=['clip_b16']  # 노트북마다 하나: ['xception'] / ['clip_b16'] / ['dinov2_b14']
STRATEGIES=['clean_only','aug_single','aug_pair','aug_pair_rank']
assert len(MODELS)==1,'동시 실행 시 노트북마다 모델 하나'
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
FF_DATA=ROOT/'data/ffpp_c23_4methods_roi16_v1'
CDF_DATA=ROOT/'data/celebdf_v2_face_crops16_v1'
SOURCE_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
TARGET_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
BASELINE_DRIVE_RUNS=ROOT/'runs/ffpp_c23_4methods_baseline_v1'
BASELINE_RUNS=Path('/content/ffpp_c23_baseline_eval_runs')
EXPERIMENT='ffpp_c23_smoothing_source_ablation'
GROUP=MODELS[0]
RUNS_ROOT=ROOT/'runs'/f'{EXPERIMENT}_{GROUP}'
MANIFEST_ROOT=ROOT/'manifests'/f'{EXPERIMENT}_{GROUP}'/'seed42'
EVALUATION_ROOT=ROOT/'evaluations'/f'{EXPERIMENT}_{GROUP}'/'seed42'
VALIDATION_CACHE=Path('/content/smoothing_mean_teacher_source_validation_v1')
TRAIN_CACHE=Path('/content/smoothing_pivot_bilateral_cache_v1')
CDF_CACHE=Path('/content/celebdf_official_smoothing_cache_v1')
CDF_PIXEL_CACHE=Path('/content/smoothing_ranking_cdf_bilateral_cache_v1')
COMMON=f'--models {GROUP} --strategies {" ".join(STRATEGIES)} --seed 42 --source-root "{SOURCE_ROOT}" --target-root "{TARGET_ROOT}" --baseline-runs-root "{BASELINE_RUNS}" --runs-root "{RUNS_ROOT}" --manifest-root "{MANIFEST_ROOT}" --output-root "{EVALUATION_ROOT}" --validation-cache "{VALIDATION_CACHE}" --train-cache "{TRAIN_CACHE}" --cache-root "{CDF_CACHE}" --evaluation-pixel-cache "{CDF_PIXEL_CACHE}" --workers 2'
print('Model:',GROUP,'| 조건:',STRATEGIES,'| 선택 = FF++ val pooled AUC',flush=True)
print('Results:',EVALUATION_ROOT,flush=True)

try:
    import googleapiclient.discovery
except ImportError:
    subprocess.run([sys.executable,'-m','pip','install','-q','google-api-python-client','google-auth'],check=True)
from google.colab import auth
auth.authenticate_user()
sys.path.insert(0,str(PROJECT/'scripts'))
import drive_folder_guard
drive_folder_guard.check([RUNS_ROOT,MANIFEST_ROOT,EVALUATION_ROOT],model_tag=GROUP)
```

## Cell 3 — FF++·초기 baseline 준비

```python
!python -u "{PROJECT}/scripts/prepare_ffpp_c23_training_data.py" --drive-data "{FF_DATA}" --data-root "{SOURCE_ROOT}" --trainval-only
assert _exit_code==0,'FF++ 준비 실패: 위 오류 확인'

name=f'{GROUP}_ffpp_c23_roi16_baseline_v1_seed42'
source,dest=BASELINE_DRIVE_RUNS/name,BASELINE_RUNS/name
dest.mkdir(parents=True,exist_ok=True)
for filename in ['best.pt','config.json','completed.json']:
    src,dst=source/filename,dest/filename
    assert src.is_file(),src
    if dst.is_file() and dst.stat().st_size==src.stat().st_size:
        continue
    print('Copying:',GROUP,filename,flush=True)
    temp=dst.with_suffix(dst.suffix+'.copying')
    shutil.copy2(src,temp)
    assert temp.stat().st_size==src.stat().st_size,'복사 미완료'
    temp.replace(dst)
```

## Cell 4 — CDF 공식 test 준비 (평가용만)

```python
name,size='celebdf_v2_roi16_test_v1.tar',557015040
manifest=TARGET_ROOT/'manifests/celebdf_v2_roi16_test_v1.csv'
if manifest.is_file():
    print('Already extracted:',name,flush=True)
else:
    source=CDF_DATA/name
    assert source.is_file() and source.stat().st_size==size,f'TAR 업로드 확인: {source}'
    local=Path('/content')/name
    print('Copying:',name,flush=True); shutil.copy2(source,local)
    assert local.stat().st_size==size,'복사 미완료'
    print('Extracting:',name,flush=True)
    subprocess.run(['tar','-xf',str(local),'-C','/content'],check=True); local.unlink()
    assert manifest.is_file(),manifest
```

## Cell 5 — 네 조건 학습

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_source_ablation.py" train {COMMON} --epochs 10 --patience 3 --source-images 6000 --val-images 1200 --batch-size 16 --encoder-lr 1e-5 --head-lr 1e-5 --ranking-weight 0.25
assert _exit_code==0,'학습 실패: 위 오류 확인 후 Cell 1부터 다시 실행하면 이어서 학습'
```

## Cell 6 — CDF 공식 test 평가

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_source_ablation.py" evaluate {COMMON} --eval-batch-size 64 --bootstrap 1000
assert _exit_code==0,'평가 실패: 위 오류 확인 후 Cell 6 재실행'
```

## Cell 7 — 결과 표시

```python
import csv
from IPython.display import Markdown,display
names={'baseline':'Baseline','clean_only':'① 원본만','aug_single':'② 스무딩 증강(한 장)','aug_pair':'③ 스무딩 증강(짝 두 장)','aug_pair_rank':'④ 스무딩 증강(짝) + 순위 손실'}
read=lambda f: list(csv.DictReader((EVALUATION_ROOT/f).open()))
rows,boot=read('summary.csv'),read('bootstrap_smoothing_gap_vs_aug_pair_rank.csv')
lines=[f'### {GROUP} — 스무딩 시프트 지표 (CDF 공식 test)','','|조건|선택 epoch|원본 AUC|진짜 쪽 격차 ↓|가짜 쪽 격차|쏠림|진짜 뒤집힘 ↓|가짜 뒤집힘 ↓|진짜 이동|가짜 이동|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"|{names[r['strategy']]}|{r['selected_epoch']}|{float(r['normal_auc']):.4f}|{float(r['smoothing_gap']):.4f}|{float(r['fake_side_gap']):.4f}|{float(r['skew']):.4f}|{float(r['flip_real@normal5']):.2%}|{float(r['flip_fake@normal5']):.2%}|{float(r['real_shift']):+.2f}σ|{float(r['fake_shift']):+.2f}σ|")
display(Markdown('\n'.join(lines)))
side={'smoothing_gap':'진짜 쪽','fake_side_gap':'가짜 쪽'}
lines=['### 격차 차이 (조건 − ④), 영상 단위 부트스트랩 95% 구간','','|조건|방향|차이|95% 구간|','|---|---|---:|---|']
for r in boot:
    lines.append(f"|{names[r['strategy']]}|{side[r['metric']]}|{float(r['observed']):+.4f}|[{float(r['ci_low']):+.4f}, {float(r['ci_high']):+.4f}]|")
display(Markdown('\n'.join(lines)))
print('차이가 양수이고 구간이 0보다 크면 ④가 그 조건보다 그 방향 격차가 작다는 뜻입니다. 불변이려면 두 방향 격차가 모두 0 근처여야 합니다.')
print('뒤집힘은 원본 진짜 FPR 5% 임계값(test 라벨 기준 진단용)에서 계산합니다. 이동은 원본 진짜 점수 표준편차 단위입니다.')
print('Results:',EVALUATION_ROOT)
```
