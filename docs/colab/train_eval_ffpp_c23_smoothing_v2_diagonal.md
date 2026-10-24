# Smoothing_V2_Diagonal (모델 공용: Xception / CLIP / DINO, 학습 + 평가)

v2 스무딩(파싱 피부 마스크 + 경계 보존 필터 4개)으로 **필터 k로 학습 → 필터 4개 모두로 평가**하는 대각선 본 실험.

- Cell 2에서 바꾸는 것은 `MODELS`(노트북마다 하나), `TRAIN_FILTERS`, `SEEDS`, `EVAL_DATASETS`뿐. 모델마다 노트북을 따로 열어 동시에 실행해도 된다(결과 폴더 첫 단계가 모델별).
- 학습 조건 4개(모두 같은 FF++ baseline seed42에서 출발, 표본·배치·선택 규칙 동일, 최대 10 epoch·patience 3):
  - `aug_single`: 원본(1/4) 또는 필터 k 스무딩 한 장 CE
  - `aug_pair`: 원본 + 스무딩 두 장 CE
  - `aug_pair_rank`: 두 장 CE + 0.25 순위 손실
  - `aug_pair_rank_fc100_ramp2` (학습법 A): `aug_pair_rank` + 100 × 특징 일관성(1−cos). 처음 2 epoch 동안 λ를 0→100으로 올리고, 이 워밍업 epoch은 patience에 세지 않는다. **세 백본 모두 같은 설정**(스크립트 기본값).
- best epoch: FF++ val 1,200장의 원본 + 필터 k 약·중·강 pooled AUC, epoch 0(Baseline)도 후보. **CDF는 학습·선택에 쓰지 않는다.** Cell 6에 선택 epoch이 나오며 0이면 그 조건이 Baseline을 넘지 못한 것.
- 평가: `EVAL_DATASETS`의 각 데이터셋에서 Baseline + 학습된 모델을 필터 4개 × 약·중·강으로 채점(대각선 = 학습 필터, 나머지 = 처음 보는 필터).
  `cdf_dev`로 먼저 전부 돌리고, **`cdf_test`(공식 test)는 모든 run이 끝난 뒤 한 번만** `EVAL_DATASETS`에 추가한다.
- 끝난 run·예측은 자동 재사용. 파일럿의 `aug_pair_rank@bilateral` seed42와 Xception `aug_pair_rank_fc100_ramp2@bilateral` seed42는 그대로 쓰인다. CLIP fc100 파일럿은 워밍업이 없어 이름이 달라(`fc100`) 재사용되지 않는다.
- λ·A'·적대 학습 파일럿 설정은 커밋 96152aa의 이 노트북에 남아 있다(코드는 그대로 지원).
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

## Cell 2 — 설정·경로

```python
MODELS=['clip_b16']                    # 노트북마다 하나: ['xception'] / ['clip_b16'] / ['dinov2_b14']
TRAIN_FILTERS=['bilateral','weighted_median','fast_global','domain_transform']
SEEDS=[42,43,44]
EVAL_DATASETS=['cdf_dev']              # 모든 run이 끝난 뒤에만 'cdf_test' 추가
assert len(MODELS)==1,'동시 실행 시 노트북마다 모델 하나'
GROUP=MODELS[0]
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
ARCHIVE=ROOT/'manifests/beauty_smoothing_v2/cache'
EXPERIMENT='ffpp_c23_smoothing_v2_diagonal'
RUNS=ROOT/'runs'/f'{EXPERIMENT}_{GROUP}'
MANIFESTS=ROOT/'manifests'/f'{EXPERIMENT}_{GROUP}'
EVAL=ROOT/'evaluations'/f'{EXPERIMENT}_{GROUP}'
BASELINE=ROOT/'runs/ffpp_c23_4methods_baseline_v1'
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
# 결과 폴더 첫 단계가 모델별이라 Drive 중복 폴더 검사(두 번째 구글 인증)는 생략. 같은 모델을 두 노트북에서 동시에 돌리지만 않으면 된다.
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

필터 하나씩 캐시를 풀어 4개 조건을 학습한다. 노트북 하나에 모델 하나 기준 seed당 필터 4 × 조건 4 = 16 run.

```python
for seed in SEEDS:
    for f in TRAIN_FILTERS:
        !python -u "{PROJECT}/scripts/ffpp_c23_smoothing_v2_diagonal.py" --source-root "{FF_ROOT}" --baseline-runs-root "{BASELINE}" --runs-root "{RUNS}" --manifest-root "{MANIFESTS}" --archive-dir "{ARCHIVE}" --train-filter {f} --models {GROUP} --seed {seed} --epochs 10 --patience 3 --source-images 6000 --val-images 1200 --batch-size 16 --workers 2
        assert _exit_code==0,f'학습 실패 (seed {seed}, {f}): 위 오류 확인 후 다시 실행'
```

## Cell 5 — 평가 (끝난 예측은 재사용)

```python
for seed in SEEDS:
    for d in EVAL_DATASETS:
        extra=f'--membership "{MEMBERSHIP}"' if d=='cdf_dev' else ''
        !python -u "{PROJECT}/scripts/evaluate_beauty_smoothing_v2.py" --dataset {d} {extra} --data-root "{DATA_ROOT[d]}" --baseline-runs-root "{BASELINE}" --archive-dir "{ARCHIVE}" --output-root "{EVAL}/seed{seed}/{d}" --models {GROUP} --runs-root "{RUNS}" --train-filters {' '.join(TRAIN_FILTERS)} --seed {seed} --eval-batch-size 64 --workers 2
        assert _exit_code==0,f'평가 실패 (seed {seed}, {d})'
```

## Cell 6 — 결과 표

각 칸 = **max|gap| (FP gap / FN gap)**. 0에 가까울수록 강건. FP gap 양수 = 진짜 오탐 방향, FN gap 양수 = 가짜 미탐 방향.
행 = 조건@학습 필터, 학습 필터 열이 대각선, 나머지 열은 처음 보는 필터. `epoch` = 선택된 epoch(0이면 Baseline이 선택됨).
마지막 표는 끝난 seed들의 평균 ± 표준편차(max|gap|, 원본 AUC).

```python
import json, sys
import pandas as pd
from IPython.display import Markdown,display
sys.path.insert(0,str(PROJECT/'scripts'))
import ffpp_c23_smoothing_v2_diagonal as diagonal
filters=['bilateral','weighted_median','fast_global','domain_transform']
def selected_epoch(seed,r):
    if r['method']=='baseline': return ''
    done=RUNS/diagonal.run_name(GROUP,r['train_filter'],r['method'],seed)/'completed.json'
    return json.loads(done.read_text())['selected_epoch'] if done.is_file() else '?'
for d in EVAL_DATASETS:
    tables=[]
    for seed in SEEDS:
        path=EVAL/f'seed{seed}'/d/'summary.csv'
        if not path.is_file(): print('아직 없음:',path); continue
        t=pd.read_csv(path).assign(seed=seed); tables.append(t)
        rows=[]
        for label,g in t.groupby('strategy',sort=False):
            g=g.set_index('filter')
            row={'조건':label,'epoch':selected_epoch(seed,g.iloc[0]),'원본 AUC':f"{g['normal_auc'].iloc[0]:.4f}"}
            for f in filters:
                r=g.loc[f]; row[f]=f"{r['max_gap']:.3f} ({r['fp_gap']:+.2f}/{r['fn_gap']:+.2f})"
            rows.append(row)
        display(Markdown(f'### {GROUP} seed{seed} {d}')); display(pd.DataFrame(rows).set_index('조건'))
    if len(tables)>1:
        t=pd.concat(tables)
        m=t.groupby(['strategy','filter'],sort=False).agg(gap=('max_gap','mean'),sd=('max_gap','std'),auc=('normal_auc','mean'),auc_sd=('normal_auc','std'),n=('seed','nunique')).reset_index()
        rows=[]
        for label,g in m.groupby('strategy',sort=False):
            g=g.set_index('filter')
            row={'조건':label,'seeds':int(g['n'].min()),'원본 AUC':f"{g['auc'].iloc[0]:.4f}±{g['auc_sd'].iloc[0]:.4f}"}
            row.update({f:f"{g.loc[f,'gap']:.3f}±{g.loc[f,'sd']:.3f}" for f in filters})
            rows.append(row)
        display(Markdown(f'### {GROUP} {d}: seed 평균 ± 표준편차 (max|gap|)')); display(pd.DataFrame(rows).set_index('조건'))
print('Results:',EVAL)
```

## Cell 7 — 보정인가 불변인가 (강도 곡선 + 흔적 프로브)

- 강도 곡선: Cell 5에서 저장된 예측만 읽는다(추론 없음, 모든 seed). 보정이면 **약** 스무딩에서 점수가 진짜 쪽(−)으로 넘어가고, 불변이면 약·중·강 모두 0 근처.
- 흔적 프로브: FF++ val 1,200장(CDF 미사용)의 원본 vs 학습 필터 강 스무딩 특징으로 "스무딩됐나" 감지 AUC. Baseline·`aug_pair_rank`·학습법 A만, seed42만(필터당 약 10분).

```python
for seed in SEEDS:
    for d in EVAL_DATASETS:
        !python -u "{PROJECT}/scripts/diagnose_v2_compensation.py" curve --eval-root "{EVAL}/seed{seed}/{d}"
        assert _exit_code==0,'강도 곡선 실패'
for f in TRAIN_FILTERS:
    !python -u "{PROJECT}/scripts/diagnose_v2_compensation.py" probe --source-root "{FF_ROOT}" --baseline-runs-root "{BASELINE}" --runs-root "{RUNS}" --archive-dir "{ARCHIVE}" --output-root "{EVAL}/trace_probe" --train-filter {f} --models {GROUP} --seed 42 --images 1200 --batch-size 64 --workers 2
    assert _exit_code==0,'흔적 프로브 실패'
```
