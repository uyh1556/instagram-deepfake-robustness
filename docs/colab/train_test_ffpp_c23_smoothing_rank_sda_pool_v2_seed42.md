# DINO_CE_Rank_SDA_Pool_v2_Seed42

**0단계:** CE+Rank를 새 선택 기준(FF++ 4조건 **pooled AUC**)으로 다시 학습한다.
**1단계:** 새로 구성한 CDF 풀(pool v2)에서 진짜:가짜 1:1 SDA를 학습하고, 실제 CDF 정답이 test 성능을 올리는지 확인한다.
기본 DINOv2 B/14. Cell 2의 MODELS만 바꾸면 CLIP·Xception도 지원한다. 새 코드를 GitHub에 push한 뒤 Cell 1부터 실행한다.

|조건|학습 내용|역할|
|---|---|---|
|CE+Rank (pooled 선택)|FF++ 실제 정답 CE + 0.25 × 진짜/가짜 순위 손실|대조군. 기존 CE+Rank와 학습은 같고 선택 기준만 다름|
|SDA 새 풀 1:1|같은 source 손실 + pool v2 adapt-train의 실제 정답 CE|1단계 핵심: "정답을 주면 좋아지는가"|
|SDA 기존 풀 1:1 (선택)|같은 source 손실 + 기존 5,000장 풀의 실제 정답 CE|비율만 맞춘 효과 분리. 빼려면 STRATEGIES에서 제거|

**pool v2 구성 (Cell 4의 prepare가 만들고 고정 저장)**
- test 영상과 같은 장면 키(파일명의 두 ID 모두)를 가진 비테스트 영상만 뺀다. 기존처럼 연결을 끝까지 따라가며 빼지 않는다.
- 가짜는 원본 장면(idA_NNNN)당 최대 3개. 진짜 영상 수와 Celeb-real:YouTube-real 비율은 공식 test의 영상 비율을 따른다.
- 장면 키로 연결된 영상 묶음 단위로 adapt-train 80% / adapt-val 20%를 나눈다. 고른 영상은 16프레임 전부 사용한다.
- 구성에 폴더·라벨 정보를 쓴다(test와 같은 분포라는 실험 설계 가정). 나중 UDA용 `adapt_train_unlabelled.csv`에는 라벨이 없다.
- 파일명 장면 키는 후보 관계이며 실제 source/driver 검증이 아니다. 같은 인물의 test 등장은 기존처럼 허용한다.

**학습·선택**
- 원래 FF++ baseline에서 각각 독립 시작. FF++ epoch당 6,000장, 배치 진짜:가짜 1:1, 원본+랜덤 약/중/강, 같은 flip.
- SDA target: 매 배치 정확히 진짜 8 / 가짜 8, 클래스 안에서 영상 균등. 원본/약/중/강 각25%, 학생 반대 flip.
  target CE 최대 가중치 0.5, 첫 2epoch 선형 증가. 최대 10epoch / patience 3 / min_delta 0.0001, LR 1e-5, BN 통계 고정.
- `best.pt`: FF++ val 1,200장 × 4조건을 한 번에 합친 pooled AUC 최대(동률이면 조건 간 최소 AUC가 큰 쪽). epoch0도 후보.
- `best_adapt_val.pt`: 같은 규칙을 CDF adapt-val(라벨 있는 비테스트)에 적용한 **진단용** 선택. 조기종료는 FF++ 기준만 쓴다.
  기존 풀 SDA는 adapt-val 영상과 학습 영상이 겹치면 이 진단 선택을 평가하지 않는다.
- 공식 CDF test는 학습·선택에 쓰지 않는다.

**평가 (Cell 6)**
- CDF 공식 test 8,288장 × 원본/약/중/강, FP32 frame. Baseline은 기존 저장 예측을 재사용한다.
- 저장: 조건별 지표, pooled AUC, 4×4 조건 간 AUC, Celeb-real/YouTube-real별 FPR·AUC, 선택 epoch,
  CE+Rank(pooled) 대비 영상 단위 부트스트랩 95% 구간(테스트 표본 변동만 반영; seed 변동 아님).
- 출력은 `ffpp_c23_smoothing_rank_sda_pool_v2`에 새로 저장하며 기존 결과를 덮어쓰지 않는다. 완료 skip/마지막 완료 epoch 재개를 지원한다.

**두 모델 동시 실행 (예: DINO 코랩 + CLIP 코랩)**
- 지원한다. 평가 결과 폴더는 모델별 최상위 폴더(`evaluations/ffpp_c23_smoothing_rank_sda_pool_v2_<모델>`)라서 두 런타임이 같은 새 폴더를 만들지 않는다.
- Cell 2가 Drive API로 결과 경로에 같은 이름의 폴더가 2개 이상 있는지 확인하고, 있으면 학습 전에 멈춘다. 처음 한 번 Google 계정 인증 창이 뜬다.
- 멈추면 출력된 폴더 ID 중 쓰지 않을 복사본의 이름을 Drive에서 바꾼 뒤 Cell 2부터 다시 실행한다.

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
assert (PROJECT/'scripts/ffpp_c23_smoothing_rank_sda_pool_v2.py').is_file(),'새 코드를 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
```

## Cell 2 — 모델·경로

```python
MODELS=['dinov2_b14']  # ['clip_b16'] 또는 ['xception']; 모델별 코랩으로 나누어 실행 가능
STRATEGIES=['ce_rank_pooled','sda_new_balanced','sda_old_balanced']  # 기존 풀 비교를 빼려면 'sda_old_balanced' 삭제
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
FF_DATA=ROOT/'data/ffpp_c23_4methods_roi16_v1'
CDF_DATA=ROOT/'data/celebdf_v2_face_crops16_v1'
SOURCE_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
TARGET_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
BASELINE_DRIVE_RUNS=ROOT/'runs/ffpp_c23_4methods_baseline_v1'
BASELINE_RUNS=Path('/content/ffpp_c23_baseline_eval_runs')
EXPERIMENT='ffpp_c23_smoothing_rank_sda_pool_v2'
GROUP='_'.join(MODELS)
RUNS_ROOT=ROOT/'runs'/EXPERIMENT
MANIFEST_ROOT=ROOT/'manifests'/EXPERIMENT/'seed42'/GROUP
EVALUATION_ROOT=ROOT/'evaluations'/f'{EXPERIMENT}_{GROUP}'/'seed42'  # 모델별 최상위 폴더: 동시 실행 시 중복 폴더 방지
HARD_MANIFEST_ROOT=ROOT/'manifests/ffpp_c23_smoothing_mean_teacher_v1/seed42'
BASELINE_PREDICTIONS=ROOT/'evaluations/ffpp_c23_smoothing_rank_adaptation_v1/seed42'
VALIDATION_CACHE=Path('/content/smoothing_mean_teacher_source_validation_v1')
TRAIN_CACHE=Path('/content/smoothing_pivot_bilateral_cache_v1')
TARGET_TRAIN_CACHE=Path('/content/smoothing_rank_sda_pool_v2_target_cache')
CDF_CACHE=Path('/content/celebdf_official_smoothing_cache_v1')
CDF_PIXEL_CACHE=Path('/content/smoothing_ranking_cdf_bilateral_cache_v1')
MODEL_ARGS=' '.join(MODELS)
STRATEGY_ARGS=' '.join(STRATEGIES)
COMMON=f'--models {MODEL_ARGS} --strategies {STRATEGY_ARGS} --seed 42 --source-root "{SOURCE_ROOT}" --target-root "{TARGET_ROOT}" --baseline-runs-root "{BASELINE_RUNS}" --runs-root "{RUNS_ROOT}" --manifest-root "{MANIFEST_ROOT}" --output-root "{EVALUATION_ROOT}" --validation-cache "{VALIDATION_CACHE}" --train-cache "{TRAIN_CACHE}" --target-train-cache "{TARGET_TRAIN_CACHE}" --cache-root "{CDF_CACHE}" --evaluation-pixel-cache "{CDF_PIXEL_CACHE}" --hard-manifest-root "{HARD_MANIFEST_ROOT}" --reference-evaluations "{BASELINE_PREDICTIONS}" --family-cap 3 --adapt-val-fraction 0.2 --workers 2'
print('Models:',MODELS,'| 학습 조건:',STRATEGIES,flush=True)
print('best.pt = FF++ 4조건 pooled AUC / best_adapt_val.pt = CDF adapt-val pooled AUC (진단)',flush=True)
print('Results:',EVALUATION_ROOT,flush=True)

# 같은 이름의 Drive 폴더가 2개 이상이면 마운트 경로가 다른 복사본을 가리킬 수 있어 여기서 멈춘다(읽기 전용 검사).
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

for model in MODELS:
    name=f'{model}_ffpp_c23_roi16_baseline_v1_seed42'
    source,dest=BASELINE_DRIVE_RUNS/name,BASELINE_RUNS/name
    dest.mkdir(parents=True,exist_ok=True)
    for filename in ['best.pt','config.json','completed.json']:
        src,dst=source/filename,dest/filename
        assert src.is_file(),src
        if dst.is_file() and dst.stat().st_size==src.stat().st_size:
            continue
        print('Copying:',model,filename,flush=True)
        temp=dst.with_suffix(dst.suffix+'.copying')
        shutil.copy2(src,temp)
        assert temp.stat().st_size==src.stat().st_size,'복사 미완료'
        temp.replace(dst)
if 'sda_old_balanced' in STRATEGIES:
    assert HARD_MANIFEST_ROOT.is_dir(),f'기존 풀 manifest 경로 확인: {HARD_MANIFEST_ROOT}'
```

## Cell 4 — CDF TAR 준비·pool v2 생성

```python
archives={
    'celebdf_v2_roi16_test_v1.tar':557015040,
    'celebdf_v2_roi16_non_test_real_v1.tar':826142720,
    'celebdf_v2_roi16_non_test_fake_v1.tar':5679564800,
}
for name,size in archives.items():
    manifest=TARGET_ROOT/'manifests'/f'{name.removesuffix(".tar")}.csv'
    markers=[Path('/content')/f'.{p}_{name}.done' for p in ['adapt','sda','sda_balance','mixed_uda','consistency_c','response','pivot','ranking','mean_teacher','delta_transfer','rank_adaptation','rank_sda_pool_v2']]
    if name=='celebdf_v2_roi16_test_v1.tar':
        markers += [Path('/content/.celebdf_official_roi16_test_v1.done'),Path('/content/.preservation_b_celebdf_official_test.done')]
    if manifest.is_file() and any(p.exists() for p in markers):
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
    (Path('/content')/f'.rank_sda_pool_v2_{name}.done').touch()
    local.unlink()

!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_rank_sda_pool_v2.py" prepare {COMMON}
assert _exit_code==0,'pool v2 생성/확인 실패: 위 오류 확인'
```

## Cell 5 — 0단계·1단계 추가학습

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_rank_sda_pool_v2.py" train {COMMON} --epochs 10 --patience 3 --source-images 6000 --val-images 1200 --target-images 5000 --batch-size 16 --encoder-lr 1e-5 --head-lr 1e-5 --ranking-weight 0.25 --target-weight 0.5 --ramp-epochs 2
assert _exit_code==0,'학습 실패: 위 오류 확인 후 Cell 5 재실행'
```

## Cell 6 — CDF 공식 test 평가·부트스트랩

```python
!python -u "{PROJECT}/scripts/ffpp_c23_smoothing_rank_sda_pool_v2.py" evaluate {COMMON} --eval-batch-size 32 --bootstrap 1000
assert _exit_code==0,'평가 실패: 위 오류 확인 후 Cell 6 재실행'
```

학습 batch16과 별도로 평가 batch32를 사용한다. FP32 frame 평가이며, 저장된 예측은 다시 실행해도 재사용한다.

## Cell 7 — 결과 표시

```python
import csv
from IPython.display import Markdown,display
names={'baseline':'Baseline','ce_rank_pooled':'CE+Rank (pooled 선택)','sda_new_balanced':'SDA 새 풀 1:1','sda_old_balanced':'SDA 기존 풀 1:1'}
label=lambda s: names[s.split('@')[0]]+(' · adapt-val 선택(진단)' if s.endswith('@adapt_val') else '')
read=lambda f: list(csv.DictReader((EVALUATION_ROOT/f).open()))
pool=json.loads((MANIFEST_ROOT/'pool_v2/summary.json').read_text())
lines=['### pool v2 구성 (영상 수 / 이미지 수)','','|구분|Celeb-real|YouTube-real|가짜|','|---|---:|---:|---:|']
for part,counts in pool['partitions'].items():
    lines.append(f"|{part}|"+'|'.join(f"{counts[k]['videos']} / {counts[k]['images']}" for k in ['Celeb-real','YouTube-real','Celeb-synthesis'])+'|')
display(Markdown('\n'.join(lines)))
mix,boot,groups,sel=read('mixture_summary.csv'),read('bootstrap_vs_ce_rank_pooled.csv'),read('subgroups.csv'),read('selections.csv')
for model in MODELS:
    lines=[f'### {model} — 4조건 요약','','|방법|pooled AUC ↑|평균 AUC ↑|강 진짜 vs 원본 가짜 AUC ↑|평균 FPR ↓|평균 FNR ↓|','|---|---:|---:|---:|---:|---:|']
    for r in mix:
        if r['model']==model:
            lines.append(f"|{label(r['strategy'])}|{float(r['pooled_auc']):.4f}|{float(r['mean_auc']):.4f}|{float(r['strong_real_vs_normal_fake_auc']):.4f}|{float(r['mean_fpr']):.2%}|{float(r['mean_fnr']):.2%}|")
    display(Markdown('\n'.join(lines)))
    lines=['### CE+Rank (pooled) 대비 차이 — 영상 단위 부트스트랩 95% 구간','','|방법|지표|차이|95% 구간|','|---|---|---:|---|']
    for r in boot:
        if r['model']==model:
            lines.append(f"|{label(r['strategy'])}|{r['metric']}|{float(r['observed']):+.4f}|[{float(r['ci_low']):+.4f}, {float(r['ci_high']):+.4f}]|")
    display(Markdown('\n'.join(lines)))
    lines=['### 진짜 출처별 FPR (원본 / 강)','','|방법|Celeb-real 원본|Celeb-real 강|YouTube-real 원본|YouTube-real 강|','|---|---:|---:|---:|---:|']
    for strategy in dict.fromkeys(r['strategy'] for r in groups if r['model']==model):
        by={r['condition']:r for r in groups if r['model']==model and r['strategy']==strategy}
        lines.append(f"|{label(strategy)}|"+'|'.join(f"{float(by[c][k]):.2%}" for k in ['fpr_Celeb-real','fpr_YouTube-real'] for c in ['normal','smooth_strong'])+'|')
    display(Markdown('\n'.join(lines)))
    for r in sel:
        if r['model']==model:
            print(label(r['strategy']),': selected epoch',r['selected_epoch'],'/ completed',r['epochs_completed'])
print('SDA는 실제 CDF 정답을 쓰는 진단(상한 확인)이며 UDA가 아닙니다. 부트스트랩은 테스트 표본 변동만 반영합니다.')
print('Results:',EVALUATION_ROOT)
```
