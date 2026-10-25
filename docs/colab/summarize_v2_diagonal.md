# V2_Diagonal_Summary (세 백본 × seed 3개 한 장 정리, GPU 불필요)

각 백본 노트북이 Drive에 저장한 `evaluations/ffpp_c23_smoothing_v2_diagonal_<모델>/seed<N>/cdf_test/summary.csv`만 읽는다(추론 없음, 1분 내외).

- **헤드라인 표**(백본 × 조건, seed 평균 ± 표준편차)
  - `clean`: 원본 AUC(4개 학습 필터 평균) — 제약
  - `diag`: 학습 필터에서의 나쁜 쪽 교차 AUC
  - `held`: 처음 보는 3개 필터의 나쁜 쪽 교차 AUC(12칸 평균) — **주 지표**
  - `DT_held`: 처음 보는 DT(세탁) 칸만
  - `all`: 16칸 평균(Baseline은 4개 필터) — 한 숫자 비교
  - `fp_flip`: 강 스무딩에서 진짜가 가짜로 뒤집히는 비율(원본 5% FPR 임계값)
  - `epoch0`: epoch 0(= Baseline)이 선택된 run 수(학습 필터 4 × seed 3 = 12 중)
  - `>rank`: 16칸(seed 평균) 중 `aug_pair_rank`보다 나쁜 쪽 교차 AUC가 높은 칸 수
- **히트맵**: 백본별 rank / fc / nsf의 4×4(행 = 학습 필터, 열 = 평가 필터) 나쁜 쪽 교차 AUC seed 평균.
- 결과 CSV: `evaluations/v2_diagonal_summary/`(헤드라인, 칸별 seed 평균).

## Cell 1 — 코드

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess
PROJECT=Path('/content/instagram_deepfake_robustness')
if PROJECT.exists():
    subprocess.run(['git','-C',str(PROJECT),'pull','--ff-only'],check=True)
else:
    subprocess.run(['git','clone','https://github.com/uyh1556/instagram-deepfake-robustness.git',str(PROJECT)],check=True)
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
OUT=ROOT/'evaluations/v2_diagonal_summary'
```

## Cell 2 — 집계

```python
!python -u "{PROJECT}/scripts/summarize_v2_diagonal.py" --eval-root "{ROOT}/evaluations" --output-root "{OUT}" --models xception clip_b16 dinov2_b14 --seeds 42 43 44
assert _exit_code==0,'집계 실패'
```

## Cell 3 — 헤드라인 표 + 히트맵

```python
import pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown
h=pd.read_csv(OUT/'v2_diagonal_headline.csv')
name={'baseline':'Baseline','aug_single':'single','aug_pair':'pair','aug_pair_rank':'rank','aug_pair_rank_fc100_ramp2':'fc','aug_pair_rank_nsf':'nsf'}
def pm(r,c): return '' if pd.isna(r[c+'_mean']) else f"{r[c+'_mean']:.3f}±{0 if pd.isna(r[c+'_sd']) else r[c+'_sd']:.3f}"
for m,g in h.groupby('model',sort=False):
    rows=[{'조건':name.get(r.method,r.method),'clean':pm(r,'clean_auc'),'diag':pm(r,'diag_worse'),'held':pm(r,'heldout_worse'),
           'DT_held':pm(r,'dt_heldout'),'all':pm(r,'all_worse'),'fp_flip':pm(r,'fp_flip'),'epoch0':int(r.epoch0_runs),
           '>rank':'' if pd.isna(r.cells_better_than_rank) else f"{int(r.cells_better_than_rank)}/16"} for r in g.itertuples()]
    display(Markdown(f'### {m} (seeds {int(g.seeds.max())})')); display(pd.DataFrame(rows).set_index('조건'))
s=pd.read_csv(OUT/'v2_diagonal_seed_means.csv')
F=['bilateral','weighted_median','fast_global','domain_transform']; S=['bil','WMF','FGS','DT']
methods=['aug_pair_rank','aug_pair_rank_fc100_ramp2','aug_pair_rank_nsf']; models=list(h.model.unique())
fig,ax=plt.subplots(len(models),3,figsize=(12,3.6*len(models)),squeeze=False)
for i,m in enumerate(models):
    for j,me in enumerate(methods):
        g=s[(s.model==m)&(s.method==me)].pivot(index='train_filter',columns='filter',values='worse_cross_auc_mean').reindex(index=F,columns=F)
        a=ax[i][j]; im=a.imshow(g.values,vmin=.45,vmax=.85,cmap='viridis')
        for r in range(4):
            for c in range(4):
                v=g.values[r,c]; a.text(c,r,'' if pd.isna(v) else f'{v:.2f}',ha='center',va='center',color='white' if v<.7 else 'black',fontsize=9)
        a.set_xticks(range(4)); a.set_xticklabels(S); a.set_yticks(range(4)); a.set_yticklabels(S)
        a.set_title(f'{m} · {name[me]}'); a.set_xlabel('평가 필터' if i==len(models)-1 else ''); a.set_ylabel('학습 필터' if j==0 else '')
fig.colorbar(im,ax=ax,shrink=.6,label='나쁜 쪽 교차 AUC (seed 평균)')
plt.show()
print('CSV:',OUT)
```
