# Rank_Weight_Check_dinov2_b14

교차 순위(`aug_pair_rank`)의 순위 손실 가중치 α를 **1, 2, 4**로 바꿔 다시 학습하고, 논문의 α = 0.25·동시 증강·제안 방법(동시 증강 + 눈·입)과 비교한다. **seed 42만**, 이 노트북은 `dinov2_b14` 전용.

- 학습: 논문과 같은 트레이너(`ffpp_c23_smoothing_v2_diagonal.py`)에서 `--ranking-weight`만 바꾼다. baseline seed42 출발, epoch 표본·배치·flip·강도 추첨, 최대 10 epoch·patience 3, best epoch(FF++ val 원본+학습 필터 약·중·강 pooled AUC, epoch 0 포함)까지 전부 논문과 같다. α마다 run 폴더가 따로라(`runs/ffpp_c23_rank_weight_v2_dinov2_b14/alpha1` 등) **논문 run은 건드리지 않는다.**
- 평가 스크립트가 자동 확인: 새 run의 설정이 논문 α=0.25 run과 **가중치만** 다른지, 같은 epoch 표본·배치를 썼는지. 논문의 동시 증강·α=0.25·제안 방법 숫자가 저장된 결과와 **똑같이 재현되는지**(아니면 중단).
- **α 선정은 Celeb-DF 비테스트(cdf_dev)로** (논문의 λ·w와 같은 방식, 규칙은 결과 전 고정): α = 0.25·1·2·4의 교차 순위 run을 cdf_dev에서 4×4(학습 필터 4 × 평가 필터 4, 강 보정)로 평가하고, 세 백본의 16칸 평균 최저 교차 AUC를 다시 평균해 가장 높은 α를 고른다. 단, 0.25보다 **0.005 이상** 높을 때만 바꾸고 아니면 0.25 유지. 세 백본 공통 α 하나.
- **공식 test 평가는 α가 정해진 뒤에만 실행된다**(Cell 7은 선정 파일이 없으면 멈춤). test에서는 원본 + 4필터 강 보정으로 모든 α를 보고하고, 모든 방법에 눈·입 점수 결합(w = 0.2, 기존 점수기, B는 탐지 모델과 같은 입력 이미지)도 같이 계산한다.
- 세 백본을 노트북 3개로 동시에 돌려도 된다(결과 폴더 첫 단계가 백본별). 끊기면 Cell 1부터 다시 실행: 끝난 run은 건너뛰고, 중단된 run은 마지막 완료 epoch부터 이어서 학습한다. Cell 4 출력에서 `RESUME`/`ALREADY COMPLETED`/`START NEW`를 확인.
- 학습량: 필터 4 × α 3 = 12 run. 런타임: GPU(L4/A100).
- 이전 버전 노트북으로 학습 중이던 run은 그대로 이어진다(학습 코드·폴더·설정 동일). 학습이 돌고 있으면 끝날 때까지 두고, 새 노트북에서 Cell 1~4를 다시 실행하면 끝난 run은 건너뛴다.

## Cell 1 — 코드 설치

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess, sys, os, shutil
PROJECT = Path('/content/instagram_deepfake_robustness')
if PROJECT.exists():
    subprocess.run(['git','-C',str(PROJECT),'pull','--ff-only'],check=True)
else:
    subprocess.run(['git','clone','https://github.com/uyh1556/instagram-deepfake-robustness.git',str(PROJECT)],check=True)
TRAIN = PROJECT/'scripts/ffpp_c23_smoothing_v2_diagonal.py'
CHECK = PROJECT/'scripts/rank_weight_check_v2.py'
assert TRAIN.is_file() and CHECK.is_file(), '새 코드를 GitHub에 push한 후 Cell 1을 다시 실행하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(), "GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
print('CPU cores:',os.cpu_count(),'| local free GB:',shutil.disk_usage('/content').free//2**30)
```

## Cell 2 — 설정·경로 (논문 결과가 모두 있는지 확인)

```python
MODEL = 'dinov2_b14'
ALPHAS = [1, 2, 4]
FILTERS = ['bilateral','weighted_median','fast_global','domain_transform']
ROOT = Path('/content/drive/MyDrive/instagram-deepfake-robustness')
EXPERIMENT = 'ffpp_c23_rank_weight_v2'
RUNS = ROOT/'runs'/f'{EXPERIMENT}_{MODEL}'            # alpha1/alpha2/alpha4 하위 폴더
MANIFESTS = ROOT/'manifests'/f'{EXPERIMENT}_{MODEL}'
OUTPUT = ROOT/'evaluations'/f'{EXPERIMENT}_{MODEL}'
THESIS_RUNS = ROOT/'runs'/f'ffpp_c23_smoothing_v2_diagonal_{MODEL}'
THESIS_EVAL = ROOT/'evaluations'/f'ffpp_c23_smoothing_v2_diagonal_{MODEL}'
PAIR_NSF = ROOT/'evaluations'/f'ffpp_c23_pair_nsf_v2_{MODEL}'
SMOOTHED_B = ROOT/'evaluations'/f'ffpp_c23_nsf_smoothed_b_v2_{MODEL}'
BASELINE = ROOT/'runs/ffpp_c23_4methods_baseline_v1'
ARCHIVE = ROOT/'manifests/beauty_smoothing_v2/cache'
FF_ROOT = Path('/content/ffpp_c23_native_roi16_v1')
CDF_ROOT = Path('/content/celebdf_v2_face_crops16_v1')
WORKERS = max(2, min(8, os.cpu_count()))
MEMBERSHIP = next((ROOT/'manifests/ffpp_c23_smoothing_rank_sda_pool_v2/seed42'/m/'pool_v2/membership_labelled.csv' for m in ['clip_b16','dinov2_b14','xception']
                   if (ROOT/'manifests/ffpp_c23_smoothing_rank_sda_pool_v2/seed42'/m/'pool_v2/membership_labelled.csv').is_file()), None)
assert MEMBERSHIP is not None, 'cdf_dev membership_labelled.csv를 찾지 못했습니다'
for f in FILTERS:
    for label in ['aug_pair','aug_pair_rank']:
        run = THESIS_RUNS/f'{MODEL}_{f}_{label}_ffpp_c23_smoothing_v2_diagonal_v1_seed42'
        assert (run/'completed.json').is_file(), f'논문 run 없음: {run}'
    assert (PAIR_NSF/'calibration'/f'seed42_{f}.json').is_file(), f'pair+NSF 결과 없음: {PAIR_NSF}'
assert (THESIS_EVAL/'seed42/cdf_test/summary.csv').is_file(), f'논문 평가 없음: {THESIS_EVAL}'
assert (THESIS_RUNS/'nonskin_probe'/f'{MODEL}_face_inner_seed42.json').is_file(), '눈·입 점수기 없음'
assert (SMOOTHED_B/'seed42/summary.csv').is_file() and all((SMOOTHED_B/'scores'/f'cdf_test_{v}.csv').is_file() for v in ['normal',*FILTERS]), f'점수 B(보정 입력) 없음: {SMOOTHED_B}'
for d in ['ffpp_trainval','cdf_dev','cdf_test']:
    for part in ['mask',*FILTERS]:
        assert (ARCHIVE/f'{d}_{part}.tar').is_file(), f'캐시 TAR 없음: {d}_{part}.tar'
GPU_ARGS = f'--baseline-runs-root "{BASELINE}" --archive-dir "{ARCHIVE}" --source-root "{FF_ROOT}" --cdf-root "{CDF_ROOT}" --batch-size 64 --workers {WORKERS}'
COMMON = (f'--model {MODEL} --alphas {" ".join(map(str,ALPHAS))} --runs-base "{RUNS}" --original-runs "{THESIS_RUNS}" '
          f'--original-eval "{THESIS_EVAL}" --pair-nsf-root "{PAIR_NSF}" --smoothed-b-root "{SMOOTHED_B}" --output-root "{OUTPUT}"')
print('Model:',MODEL,'| alphas:',ALPHAS,'| runs:',RUNS,'| workers:',WORKERS,flush=True)
```

## Cell 3 — 데이터 준비 (FF++ train/val, CDF test, CDF 비테스트)

```python
!python -u "{PROJECT}/scripts/prepare_ffpp_c23_training_data.py" --drive-data "{ROOT}/data/ffpp_c23_4methods_roi16_v1" --data-root "{FF_ROOT}" --trainval-only
assert _exit_code==0, 'FF++ train/val 준비 실패'
def extract_cdf(name, size):
    if (CDF_ROOT/'manifests'/f'{name.removesuffix(".tar")}.csv').is_file(): return
    source = ROOT/'data/celebdf_v2_face_crops16_v1'/name
    assert source.is_file() and source.stat().st_size==size, f'TAR 확인: {source}'
    assert shutil.disk_usage('/content').free > 2*size+2*2**30, 'Colab 로컬 공간 부족'
    local = Path('/content')/name; print('Copying:',name,flush=True); shutil.copy2(source,local)
    subprocess.run(['tar','-xf',str(local),'-C','/content'],check=True); local.unlink()
extract_cdf('celebdf_v2_roi16_test_v1.tar', 557015040)           # 공식 test (dev 단계의 test 영상 겹침 검사에도 필요)
extract_cdf('celebdf_v2_roi16_non_test_real_v1.tar', 826142720)   # cdf_dev
extract_cdf('celebdf_v2_roi16_non_test_fake_v1.tar', 5679564800)
print('Data ready', flush=True)
```

## Cell 4 — 학습 (필터 4 × α 3 = 12 run)

필터마다 학습 캐시를 한 번 풀고 α 1·2·4를 차례로 학습한다. 끊기면 다시 실행: 끝난 run은 `ALREADY COMPLETED`, 중단된 run은 `RESUME ... from completed epoch N`으로 이어진다.

```python
for f in FILTERS:
    for a in ALPHAS:
        tag = f'alpha{a:g}'
        print(f'===== {MODEL} | {f} | {tag} =====', flush=True)
        !python -u "{TRAIN}" --source-root "{FF_ROOT}" --baseline-runs-root "{BASELINE}" --runs-root "{RUNS}/{tag}" --manifest-root "{MANIFESTS}" --archive-dir "{ARCHIVE}" --train-filter {f} --models {MODEL} --strategies aug_pair_rank --ranking-weight {a} --seed 42 --epochs 10 --patience 3 --source-images 6000 --val-images 1200 --batch-size 16 --workers 2
        assert _exit_code==0, f'학습 실패 ({f}, alpha {a}): 위 오류 확인 후 다시 실행'
```

## Cell 5 — α 선정용 평가 (Celeb-DF 비테스트 cdf_dev, 4×4, 끝난 예측은 재사용)

α = 0.25·1·2·4 각각 학습 필터 4 × 평가 필터 4 = 16칸(강 보정)을 평가한다. 공식 test는 쓰지 않는다. 세 백본의 dev 결과가 모두 모이면 α를 자동으로 고른다(먼저 끝난 백본은 "Waiting ..."이 나오면 정상, 나머지 노트북이 Cell 5를 마친 뒤 Cell 5를 다시 실행).

```python
!python -u "{CHECK}" dev {COMMON} --membership "{MEMBERSHIP}" {GPU_ARGS}
assert _exit_code==0, 'cdf_dev 평가 실패: 위 오류 확인 후 다시 실행'
```

## Cell 6 — α 선정 결과 (세 백본 공통)

```python
import json
import pandas as pd
from IPython.display import display, Markdown
dev_table = pd.read_csv(OUTPUT/'dev/dev_headline.csv')
display(Markdown(f'### {MODEL} — cdf_dev 최저 교차 AUC (seed 42)'))
display(dev_table.rename(columns={'alpha':'α','clean_auc':'원본 AUC','seen4':'학습한 4','unseen12':'처음 보는 12','all16':'전체 16'}).set_index('α').round(4))
sel = OUTPUT/'alpha_selection.json'
if sel.is_file():
    d = json.loads(sel.read_text())
    display(Markdown(f"### 선정된 α = **{d['chosen_alpha']:g}** (최고 {d['best_alpha']:g}, 0.25보다 {d['margin']} 이상 높아야 교체)"))
    display(pd.DataFrame(d['per_backbone']).assign(평균=pd.Series(d['scores'])).round(4))
else:
    print('아직 세 백본의 cdf_dev 결과가 모이지 않았습니다. 다른 노트북의 Cell 5가 끝나면 Cell 5를 다시 실행하세요.')
```

## Cell 7 — 공식 test 평가 (α 선정 후에만 실행됨; 원본 + 강 보정 4종, FF++ val 통계, 눈·입 결합)

```python
assert (OUTPUT/'alpha_selection.json').is_file(), 'α가 아직 선정되지 않았습니다: 세 백본 모두 Cell 5를 먼저 끝내세요'
!python -u "{CHECK}" collect {COMMON} {GPU_ARGS}
assert _exit_code==0, 'test 평가 실패: 위 오류 확인 후 다시 실행'
```

## Cell 8 — test 결과 표 (seed 42, 강 보정 최저 교차 AUC)

- 학습한 4 / 처음 보는 12 / 전체 16 = 학습한 스무딩 4조건, 학습하지 않은 스무딩 12조건, 전체 16조건 평균
- 동시 증강보다 높은 칸 / 제안 방법보다 높은 칸 = 16칸 중 개수
- α·rank÷CE = 학습 중 평균 순위 손실 × α ÷ 평균 분류 손실 (순위 손실의 비중)
- dev 선정 = cdf_dev로 고른 α의 행

```python
h = pd.read_csv(OUTPUT/'rank_weight_headline.csv')
table = h.rename(columns={'name':'방법','clean_auc':'원본 AUC','seen4':'학습한 4','unseen12':'처음 보는 12','all16':'전체 16',
                          'cells_over_pair':'동시 증강보다 높은 칸','cells_over_proposed':'제안 방법보다 높은 칸',
                          'selected_epochs':'선택 epoch (Bil/WMF/FGS/DT)','weighted_rank_over_ce':'α·rank÷CE','dev_selected':'dev 선정'})
cols = ['방법','원본 AUC','학습한 4','처음 보는 12','전체 16','동시 증강보다 높은 칸','제안 방법보다 높은 칸','선택 epoch (Bil/WMF/FGS/DT)','α·rank÷CE','dev 선정']
display(Markdown(f'### {MODEL} — seed 42, Celeb-DF v2 공식 test'))
display(table[cols].set_index('방법').round(4))
display(Markdown('### 학습 손실 (run별 전체 epoch 평균)'))
display(pd.read_csv(OUTPUT/'rank_weight_training.csv').round(4))
print('Saved:', OUTPUT)
```
