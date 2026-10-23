# Smoothing_V2_Diagonal (모델 공용: Xception / CLIP / DINO, 학습 + 평가)

v2 스무딩(파싱 피부 마스크 + 경계 보존 필터 4개)으로 **필터 k로 학습 → 필터 4개 모두로 평가**하는 대각선 실험. 파일럿과 본 실험이 같은 노트북·같은 스크립트를 쓰므로 파일럿 run은 그대로 본 실험에 재사용된다(끝난 run은 자동으로 건너뜀).

- Cell 2의 `MODELS`에 모델 **하나**. 모델마다 노트북을 따로 열어 동시에 실행해도 된다(결과 폴더 첫 단계가 모델별).
- 학습 조건(모두 같은 FF++ baseline seed42에서 출발, 표본·배치·선택 규칙 동일, 최대 10 epoch·patience 3):
  - `aug_single`: 원본(1/4) 또는 필터 k 스무딩 한 장 CE
  - `aug_pair`: 원본 + 스무딩 두 장 CE
  - `aug_pair_rank`: 두 장 CE + 0.25 순위 손실
  - `aug_pair_rank_fc` (**학습법 A**): `aug_pair_rank` + 1.0 × 특징 일관성(스무딩 특징을 원본 특징 쪽으로 당김, 원본은 고정점)
- best epoch: FF++ val 1,200장의 원본 + 필터 k 약·중·강 pooled AUC. **CDF는 학습·선택에 쓰지 않는다.**
- 평가: `EVAL_DATASETS`의 각 데이터셋에서 Baseline + 학습된 모델을 필터 4개 × 약·중·강으로 채점(대각선 = 학습 필터, 나머지 = 처음 보는 필터).
  방법 개발(파일럿) 동안은 `cdf_dev`(CDF 비테스트)와 `ffpp_test`만 본다. **`cdf_test`(공식 test)는 방법을 확정한 뒤 본 실험에서 한 번만.**
- **파일럿(λ 정하기):** `MODELS`=CLIP 또는 Xception, `TRAIN_FILTERS=['bilateral']`, `SEEDS=[42]`, `STRATEGIES=['aug_pair_rank','aug_pair_rank_fc']`,
  `FC_WEIGHTS=[1,10,100]`(λ마다 run 하나: `aug_pair_rank_fc1/10/100`), `EVAL_DATASETS=['cdf_dev']`. 백본당 4 run.
- **본 실험:** 고른 λ 하나로 `FC_WEIGHTS=[λ]`, 전략 4개, `TRAIN_FILTERS` 4개, `SEEDS=[42,43,44]`, 세 백본, 마지막에 `cdf_test` 추가. 파일럿 run은 그대로 재사용.
- 런타임: GPU(L4/A100). 학습용 캐시는 학습 필터와 마스크 tar만 푼다(필터 하나 약 9GB).

## Cell 1 — 설치·코드 갱신

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess, sys, shutil, os

PROJECT=Path('/content/instagram_deepfake_robustness')
if PROJECT.exists():
    subprocess.run(['git','-C',str(PROJECT),'pull','--ff-only'],check=True)
else:
    subprocess.run(['git','clone','https://github.com/uyh1556/instagram-deepfake-robustness.git',str(PROJECT)],check=True)
assert (PROJECT/'scripts/ffpp_c23_smoothing_v2_diagonal.py').is_file(),'새 코드를 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
print('CPU cores:',os.cpu_count(),'| local free GB:',shutil.disk_usage('/content').free//2**30)
```

## Cell 2 — 설정·경로·중복 폴더 검사

```python
MODELS=['clip_b16']                    # 노트북마다 하나: ['xception'] / ['clip_b16'] / ['dinov2_b14']
TRAIN_FILTERS=['bilateral']            # 본 실험: ['bilateral','weighted_median','fast_global','domain_transform']
STRATEGIES=['aug_pair_rank','aug_pair_rank_fc']   # 본 실험: ['aug_single','aug_pair','aug_pair_rank','aug_pair_rank_fc']
FC_WEIGHTS=[1,10,100]                  # 일관성 λ 후보 (본 실험: 고른 λ 하나)
SEEDS=[42]                             # 본 실험: [42,43,44]
EVAL_DATASETS=['cdf_dev']              # 공식 test 'cdf_test'는 방법 확정 후에만 추가
assert len(MODELS)==1,'동시 실행 시 노트북마다 모델 하나'
GROUP=MODELS[0]
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
ARCHIVE=ROOT/'manifests/beauty_smoothing_v2/cache'
EXPERIMENT='ffpp_c23_smoothing_v2_diagonal'
RUNS=ROOT/'runs'/f'{EXPERIMENT}_{GROUP}'
MANIFESTS=ROOT/'manifests'/f'{EXPERIMENT}_{GROUP}'
EVAL=ROOT/'evaluations'/f'{EXPERIMENT}_{GROUP}'
FF_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
CDF_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
DATA_ROOT={'cdf_test':CDF_ROOT,'cdf_dev':CDF_ROOT,'ffpp_test':FF_ROOT}
MEMBERSHIP=next((ROOT/'manifests/ffpp_c23_smoothing_rank_sda_pool_v2/seed42'/m/'pool_v2/membership_labelled.csv' for m in ['clip_b16','dinov2_b14','xception']
                 if (ROOT/'manifests/ffpp_c23_smoothing_rank_sda_pool_v2/seed42'/m/'pool_v2/membership_labelled.csv').is_file()),None)
for d in ['ffpp_trainval',*EVAL_DATASETS]:
    parts=['mask',*TRAIN_FILTERS] if d=='ffpp_trainval' else ['mask','bilateral','weighted_median','fast_global','domain_transform']
    for part in parts:
        assert (ARCHIVE/f'{d}_{part}.tar').is_file(),f'캐시 노트북을 먼저 실행하세요: {d}_{part}.tar'
assert 'cdf_dev' not in EVAL_DATASETS or MEMBERSHIP is not None,'pool v2 membership_labelled.csv를 찾지 못했습니다'
print('Model:',GROUP,'| filters:',TRAIN_FILTERS,'| seeds:',SEEDS,'| eval:',EVAL_DATASETS,flush=True)

try:
    import googleapiclient.discovery
except ImportError:
    subprocess.run([sys.executable,'-m','pip','install','-q','google-api-python-client','google-auth'],check=True)
from google.colab import auth
auth.authenticate_user()
sys.path.insert(0,str(PROJECT/'scripts'))
import drive_folder_guard
drive_folder_guard.check([RUNS,MANIFESTS,EVAL],model_tag=GROUP)
```

## Cell 3 — 데이터 준비 (FF++ train/val·test, CDF)

```python
!python -u "{PROJECT}/scripts/prepare_ffpp_c23_training_data.py" --drive-data "{ROOT}/data/ffpp_c23_4methods_roi16_v1" --data-root "{FF_ROOT}" --trainval-only
assert _exit_code==0,'FF++ train/val 준비 실패'
if 'ffpp_test' in EVAL_DATASETS:
    !python -u "{PROJECT}/scripts/prepare_ffpp_c23_training_data.py" --drive-data "{ROOT}/data/ffpp_c23_4methods_roi16_v1" --data-root "{FF_ROOT}" --test-only
    assert _exit_code==0,'FF++ test 준비 실패'

def extract_cdf(name,size):
    if (CDF_ROOT/'manifests'/f'{name.removesuffix(".tar")}.csv').is_file(): return
    source=ROOT/'data/celebdf_v2_face_crops16_v1'/name
    assert source.is_file() and source.stat().st_size==size,f'TAR 확인: {source}'
    assert shutil.disk_usage('/content').free>2*size+2*2**30,'Colab 로컬 공간 부족'
    local=Path('/content')/name; print('Copying:',name,flush=True); shutil.copy2(source,local)
    subprocess.run(['tar','-xf',str(local),'-C','/content'],check=True); local.unlink()
if {'cdf_test','cdf_dev'} & set(EVAL_DATASETS):
    extract_cdf('celebdf_v2_roi16_test_v1.tar',557015040)
if 'cdf_dev' in EVAL_DATASETS:
    extract_cdf('celebdf_v2_roi16_non_test_real_v1.tar',826142720)
    extract_cdf('celebdf_v2_roi16_non_test_fake_v1.tar',5679564800)
```

## Cell 4 — 학습 (끊기면 다시 실행: 끝난 run은 건너뛰고, 중단된 run은 마지막 완료 epoch부터)

```python
for seed in SEEDS:
    for f in TRAIN_FILTERS:
        !python -u "{PROJECT}/scripts/ffpp_c23_smoothing_v2_diagonal.py" --source-root "{FF_ROOT}" --baseline-runs-root "{ROOT}/runs/ffpp_c23_4methods_baseline_v1" --runs-root "{RUNS}" --manifest-root "{MANIFESTS}" --archive-dir "{ARCHIVE}" --train-filter {f} --models {GROUP} --strategies {' '.join(STRATEGIES)} --fc-weights {' '.join(map(str,FC_WEIGHTS))} --seed {seed} --epochs 10 --patience 3 --source-images 6000 --val-images 1200 --batch-size 16 --workers 2
        assert _exit_code==0,f'학습 실패 (seed {seed}, {f}): 위 오류 확인 후 다시 실행'
```

## Cell 5 — 평가 (끝난 예측은 재사용)

```python
for seed in SEEDS:
    for d in EVAL_DATASETS:
        extra=f'--membership "{MEMBERSHIP}"' if d=='cdf_dev' else ''
        !python -u "{PROJECT}/scripts/evaluate_beauty_smoothing_v2.py" --dataset {d} {extra} --data-root "{DATA_ROOT[d]}" --baseline-runs-root "{ROOT}/runs/ffpp_c23_4methods_baseline_v1" --archive-dir "{ARCHIVE}" --output-root "{EVAL}/seed{seed}/{d}" --models {GROUP} --runs-root "{RUNS}" --train-filters {' '.join(TRAIN_FILTERS)} --strategies {' '.join(STRATEGIES)} --fc-weights {' '.join(map(str,FC_WEIGHTS))} --seed {seed} --eval-batch-size 64 --workers 2
        assert _exit_code==0,f'평가 실패 (seed {seed}, {d})'
```

## Cell 6 — 결과 표 (한 장)

각 칸 = **max|gap| (FP gap / FN gap)**. max|gap|이 0에 가까울수록 강건. FP gap 양수 = 보정된 진짜 오탐 방향, FN gap 양수 = 보정된 가짜 미탐 방향.
학습 필터 열이 학습한 상황, 나머지 열은 처음 보는 필터. 원본 AUC는 스무딩 없는 탐지 실력(떨어지면 안 됨).

```python
import pandas as pd
from IPython.display import Markdown,display
filters=['bilateral','weighted_median','fast_global','domain_transform']
for seed in SEEDS:
    for d in EVAL_DATASETS:
        t=pd.read_csv(EVAL/f'seed{seed}'/d/'summary.csv')
        rows=[]
        for label,g in t.groupby('strategy',sort=False):
            g=g.set_index('filter')
            row={'조건':label,'원본 AUC':f"{g['normal_auc'].iloc[0]:.4f}"}
            for f in filters:
                r=g.loc[f]; row[f]=f"{r['max_gap']:.3f} ({r['fp_gap']:+.2f}/{r['fn_gap']:+.2f})"
            rows.append(row)
        display(Markdown(f'### {GROUP} seed{seed} {d}')); display(pd.DataFrame(rows).set_index('조건'))
print('Results:',EVAL)
```
