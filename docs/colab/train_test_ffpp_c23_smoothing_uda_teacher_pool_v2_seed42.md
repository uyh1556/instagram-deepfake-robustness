# UDA_Teacher_Pool_v2_Seed42 (모델 공용: Xception / CLIP / DINO)

교사가 학생의 실수를 물려받는 악순환을 막는 두 방법을, **조기종료 없이 고정 6 epoch**로 비교한다.
Cell 2의 `MODELS`에 모델 **하나**를 넣는다. 모델마다 이 노트북을 따로 열어 **동시에** 실행해도 된다.

|조건|교사|임시 정답이 학습시키는 곳|
|---|---|---|
|`ce_rank_fixed`|없음 (CDF 미사용)|– · 같은 6 epoch 일정의 source-only 대조군|
|`ts_ema` (TS 기본)|학생의 EMA(0.999)|판정 head (기존 TS와 같음)|
|`ts_frozen_teacher`|FF++ baseline 그대로, **업데이트 안 함**|판정 head|
|`ts_decoupled_head`|학생 인코더+**판정 head**의 EMA|**보조 head만** (판정 head는 FF++ 정답만)|

- 네 조건 모두 원래 FF++ baseline에서 독립 시작. source 손실은 CE+Rank(원본+스무딩 짝, CE + 0.25 × 순위 손실), FF++ 6,000장/epoch.
- CDF: pool v2 adapt-train을 **라벨 없이** 영상 균등 6,000회/epoch, 원본/약/중/강 각25%. 임시 정답은 교사 확신 ≥ 0.9.
- target 가중치 최대 0.5, 처음 2 epoch 선형 증가 → 최대 가중치로 4 epoch 더 = **총 6 epoch 고정**. 결과를 보기 전에 정한 일정이다.
- **주 결과 = 6 epoch 마지막 모델**(`final.pt`). 비교용으로 FF++ val pooled AUC 최고 epoch도 `best_ffpp.pt`(`@ffpp`)로 저장한다.
- 보조 head는 판정 head의 복사본에서 시작하며 test 판정에는 쓰지 않는다.
- CDF 라벨은 학습·선택에 쓰지 않는다. 기록용으로만 읽는다: epoch별 교사 임시 정답 정확도, 교사 오탐(진짜→"가짜")·미탐(가짜→"진짜") 수, CDF adapt-val 지표(보조 head 포함).
- 라벨 없는 검증 지표도 매 epoch 기록만 한다: 예측 엔트로피, 교사–학생 일치도, SND. 나중에 어떤 지표가 믿을 만한지 사후 분석용이다.
- 평가: CDF 공식 test 8,288장 × 4조건. pooled AUC, 조건 간 AUC, Celeb-real/YouTube-real별 FPR, 나온/안 나온 인물별 AUC,
  간격 회복률 =(방법 − ce_rank_fixed)/(SDA 새 풀 − ce_rank_fixed), 부트스트랩(각 TS vs ce_rank_fixed, 두 개선안 vs ts_ema).
  이전 실험의 CE+Rank(pooled)·SDA 새 풀·TS 확신 결과는 `(ref)`로 표시해 저장 예측만 읽는다.

**동시 실행·중단 대비**
- 결과 폴더의 첫 새 단계가 모델별이다: `runs|manifests|evaluations/ffpp_c23_smoothing_uda_teacher_pool_v2_<모델>/`.
- Cell 2가 Drive API로 같은 이름 폴더가 2개 이상인지 확인하고 있으면 멈춘다(처음 한 번 Google 인증).
- epoch마다 학생·교사·보조 head·optimizer·기록을 Drive에 저장한다. 끊기면 Cell 1부터 다시 실행하면 마지막 완료 epoch부터 이어 학습하고 완료된 조건은 건너뛴다.
  학습 시작 시 `START NEW` / `RESUME from completed epoch k/6` / `ALREADY COMPLETED`가 크게 출력된다. 이전에 학습하던 run인데 `START NEW`가 보이면 멈추고 경로를 확인한다.

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
assert (PROJECT/'scripts/ffpp_c23_smoothing_uda_teacher_pool_v2.py').is_file(),'새 코드를 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
```

## Cell 2 — 모델·경로·중복 폴더 검사

```python
MODELS=['clip_b16']  # 노트북마다 하나: ['xception'] / ['clip_b16'] / ['dinov2_b14']
STRATEGIES=['ce_rank_fixed','ts_ema','ts_frozen_teacher','ts_decoupled_head']
assert len(MODELS)==1,'동시 실행 시 노트북마다 모델 하나'
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
FF_DATA=ROOT/'data/ffpp_c23_4methods_roi16_v1'
CDF_DATA=ROOT/'data/celebdf_v2_face_crops16_v1'
SOURCE_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
TARGET_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
BASELINE_DRIVE_RUNS=ROOT/'runs/ffpp_c23_4methods_baseline_v1'
BASELINE_RUNS=Path('/content/ffpp_c23_baseline_eval_runs')
EXPERIMENT='ffpp_c23_smoothing_uda_teacher_pool_v2'
GROUP=MODELS[0]
RUNS_ROOT=ROOT/'runs'/f'{EXPERIMENT}_{GROUP}'
MANIFEST_ROOT=ROOT/'manifests'/f'{EXPERIMENT}_{GROUP}'/'seed42'
EVALUATION_ROOT=ROOT/'evaluations'/f'{EXPERIMENT}_{GROUP}'/'seed42'
POOL_REFERENCE=ROOT/'manifests/ffpp_c23_smoothing_rank_sda_pool_v2/seed42'/GROUP/'pool_v2/summary.json'
REFERENCE_ROOTS=[p for p in [ROOT/'evaluations/ffpp_c23_smoothing_rank_sda_pool_v2/seed42',
                             ROOT/'evaluations'/f'ffpp_c23_smoothing_rank_sda_pool_v2_{GROUP}'/'seed42',
                             ROOT/'evaluations'/f'ffpp_c23_smoothing_rank_uda_pool_v2_{GROUP}'/'seed42',
                             ROOT/'evaluations/ffpp_c23_smoothing_rank_adaptation_v1/seed42',
                             ROOT/'evaluations/ffpp_c23_smoothing_delta_transfer_v1/seed42'] if p.is_dir()]
VALIDATION_CACHE=Path('/content/smoothing_mean_teacher_source_validation_v1')
TRAIN_CACHE=Path('/content/smoothing_pivot_bilateral_cache_v1')
TARGET_TRAIN_CACHE=Path('/content/smoothing_rank_sda_pool_v2_target_cache')
CDF_CACHE=Path('/content/celebdf_official_smoothing_cache_v1')
CDF_PIXEL_CACHE=Path('/content/smoothing_ranking_cdf_bilateral_cache_v1')
REFERENCE_ARGS=' '.join(f'"{p}"' for p in REFERENCE_ROOTS)
POOL_ARG=f'--pool-reference "{POOL_REFERENCE}"' if POOL_REFERENCE.is_file() else ''
COMMON=f'--models {GROUP} --strategies {" ".join(STRATEGIES)} --seed 42 --source-root "{SOURCE_ROOT}" --target-root "{TARGET_ROOT}" --baseline-runs-root "{BASELINE_RUNS}" --runs-root "{RUNS_ROOT}" --manifest-root "{MANIFEST_ROOT}" --output-root "{EVALUATION_ROOT}" --validation-cache "{VALIDATION_CACHE}" --train-cache "{TRAIN_CACHE}" --target-train-cache "{TARGET_TRAIN_CACHE}" --cache-root "{CDF_CACHE}" --evaluation-pixel-cache "{CDF_PIXEL_CACHE}" --reference-roots {REFERENCE_ARGS} {POOL_ARG} --family-cap 3 --adapt-val-fraction 0.2 --workers 2'
print('Model:',GROUP,'| 조건:',STRATEGIES,'| 고정 6 epoch, 주 결과 = 마지막 epoch',flush=True)
print('pool v2 구성 확인:', 'CE+Rank/SDA manifest와 대조' if POOL_ARG else '참조 manifest 없음(같은 seed로 재생성)',flush=True)
print('비교용 저장 예측 경로:',[str(p) for p in REFERENCE_ROOTS],flush=True)
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

## Cell 4 — CDF TAR 준비·pool v2 확인

```python
archives={
    'celebdf_v2_roi16_test_v1.tar':557015040,
    'celebdf_v2_roi16_non_test_real_v1.tar':826142720,
    'celebdf_v2_roi16_non_test_fake_v1.tar':5679564800,
}
for name,size in archives.items():
    manifest=TARGET_ROOT/'manifests'/f'{name.removesuffix(".tar")}.csv'
    marker=Path('/content')/f'.cdf_extracted_{name}.done'
    if manifest.is_file() and (marker.exists() or any(Path('/content').glob(f'.*_{name}.done'))):
        print('Already extracted:',name,flush=True)
        continue
    source=CDF_DATA/name
    assert source.is_file() and source.stat().st_size==size,f'TAR 업로드 확인: {source}'
    assert shutil.disk_usage('/content').free>2*size+2*2**30,'Colab 로컬 공간 부족'
    local=Path('/content')/name
    print('Copying:',name,flush=True)
    shutil.copy2(source,local)
    assert local.stat().st_size==size,'복사 미완료'
    print('Extracting:',name,flush=True)
    result=subprocess.run(['tar','-xf',str(local),'-C','/content'],capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-5000:])
    assert manifest.is_file(),manifest
    marker.touch()
    local.unlink()

!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_uda_teacher_pool_v2.py" prepare {COMMON}
assert _exit_code==0,'pool v2 확인 실패: 위 오류 확인'
```

## Cell 5 — 네 조건 학습 (각 6 epoch 고정)

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_uda_teacher_pool_v2.py" train {COMMON} --epochs 6 --source-images 6000 --val-images 1200 --target-images 5000 --batch-size 16 --encoder-lr 1e-5 --head-lr 1e-5 --ranking-weight 0.25 --target-weight 0.5 --ramp-epochs 2 --ema-decay 0.999 --pseudo-threshold 0.9
assert _exit_code==0,'학습 실패: 위 오류 확인 후 Cell 1부터 다시 실행하면 이어서 학습'
```

## Cell 6 — CDF 공식 test 평가·비교

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_uda_teacher_pool_v2.py" evaluate {COMMON} --eval-batch-size 32 --bootstrap 1000
assert _exit_code==0,'평가 실패: 위 오류 확인 후 Cell 6 재실행'
```

마지막 epoch와 `@ffpp` 두 checkpoint를 모두 평가한다. 예측은 조건별로 Drive에 저장되며 다시 실행하면 재사용한다.

## Cell 7 — 결과 표시

```python
import csv
from IPython.display import Markdown,display
base={'baseline':'Baseline','ce_rank_fixed':'CE+Rank 6ep (대조군)','ts_ema':'TS 기본 6ep','ts_frozen_teacher':'TS 고정 교사 6ep',
      'ts_decoupled_head':'TS head 분리 6ep','ce_rank_pooled(ref)':'(이전) CE+Rank 조기종료','sda_new_balanced(ref)':'(이전) SDA 새 풀 상한',
      'ts_confidence(ref)':'(이전) TS 확신 조기종료'}
label=lambda s: base.get(s.split('@')[0],s)+(' · FF++ 선택' if s.endswith('@ffpp') else '')
read=lambda f: list(csv.DictReader((EVALUATION_ROOT/f).open()))
num=lambda r,k,fmt: format(float(r[k]),fmt) if r.get(k) not in (None,'') else '–'
mix,boot,pseudo,sel=read('mixture_summary.csv'),read('bootstrap.csv'),read('pseudo_labels.csv'),read('selections.csv')
lines=[f'### {GROUP} — 4조건 요약 (주 결과 = 6 epoch 마지막 모델)','','|방법|pooled AUC ↑|강 진짜 vs 원본 가짜 ↑|FPR ↓|FNR ↓|간격 회복률|나온 인물|안 나온 인물|','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in mix:
    lines.append(f"|{label(r['strategy'])}|{num(r,'pooled_auc','.4f')}|{num(r,'strong_real_vs_normal_fake_auc','.4f')}|{num(r,'mean_fpr','.2%')}|{num(r,'mean_fnr','.2%')}|{num(r,'gap_closure_vs_sda','.0%')}|{num(r,'seen_identity_pooled_auc','.4f')}|{num(r,'unseen_identity_pooled_auc','.4f')}|")
display(Markdown('\n'.join(lines)))
lines=['### 차이 — 영상 단위 부트스트랩 95% 구간 (주 결과끼리)','','|방법|기준|지표|차이|95% 구간|','|---|---|---|---:|---|']
for r in boot:
    ci=f"[{float(r['ci_low']):+.4f}, {float(r['ci_high']):+.4f}]" if r['ci_low'] else '–'
    lines.append(f"|{label(r['strategy'])}|{label(r['reference'])}|{r['metric']}|{float(r['observed']):+.4f}|{ci}|")
display(Markdown('\n'.join(lines)))
lines=['### epoch별 교사 임시 정답·검증 (라벨은 기록용)','','|방법|epoch|정확도|"가짜"/"진짜" 채택|교사 오탐|교사 미탐|FF++ pooled|adapt-val pooled|보조 head adapt-val|엔트로피|SND|교사-학생 일치|','|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in pseudo:
    lines.append(f"|{label(r['strategy'])}|{r['epoch']}|{num(r,'pseudo_accuracy','.1%')}|{r['selected_fake']}/{r['selected_real']}|{r['teacher_false_fake']}|{r['teacher_false_real']}|{num(r,'ffpp_pooled_auc','.4f')}|{num(r,'adapt_val_pooled_auc','.4f')}|{num(r,'adapt_val_aux_pooled_auc','.4f')}|{num(r,'free_mean_entropy','.3f')}|{num(r,'free_snd','.3f')}|{num(r,'free_teacher_student_agreement','.3f')}|")
display(Markdown('\n'.join(lines)))
for r in sel:
    print(label(r['strategy']),': epoch',r['epoch'],'/',r['epochs_completed'])
print('UDA는 CDF 라벨을 학습·선택에 쓰지 않습니다. (ref)는 이전 실험 저장 예측입니다.')
print('"안 나온 인물" 결과는 영상 수가 적어 경향 확인용입니다. 부트스트랩은 테스트 표본 변동만 반영합니다.')
print('Results:',EVALUATION_ROOT)
```
