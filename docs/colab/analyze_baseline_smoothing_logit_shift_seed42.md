# Baseline_Smoothing_Logit_Shift_Seed42

**CPU 런타임으로 충분. GPU·이미지/TAR·checkpoint·재학습·재추론 모두 필요 없다.**
이미 완료된 세 baseline의 원본/스무딩 예측 CSV 12개와 작은 JSON만 읽는다.

- 이미지 경로로 원본과 스무딩을 정확히 연결한다. 실제/가짜는 따로 분석한다.
- binary softmax 확률로부터 `m=log(p/(1-p))`를 복원한다. `Δm=m_smoothing−m_original`.
- Δm>0이면 가짜 쪽 이동, Δm<0이면 실제 쪽 이동. 원래 raw logits를 직접 저장한 분석은 아니다.
- **확률이 정확히0/1인 pair는 margin 분석에서 제외**하고 개수/비율을 표시한다. 임의 clipping 값으로 로짓을 만들어내지 않는다.
- 전체 이미지·검출 성공·fallback을 각각 요약한다. 그림은 **검출 성공 이미지**만 표시한다.
- 가짜 쪽 확률 이동 비율은 전체 pair, 로짓 이동 중앙값은 복원 가능한 pair 기준이므로 분모가 다를 수 있다.
- 이미지들은 영상 내에서 연관돼 있다. 이 분석은 기술통계이며 독립적인8,288개 관측의 통계적 유의성을 주장하지 않는다.
- 기존 평가 CSV/JSON은 변경하지 않고 별도 diagnostics 폴더에 결과를 저장한다.

## Cell 1 — Drive·코드

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess,sys
PROJECT=Path('/content/instagram_deepfake_robustness')
if PROJECT.exists():
    subprocess.run(['git','-C',str(PROJECT),'pull','--ff-only'],check=True)
else:
    subprocess.run(['git','clone','https://github.com/uyh1556/instagram-deepfake-robustness.git',str(PROJECT)],check=True)
assert (PROJECT/'scripts/analyze_smoothing_logit_shift.py').is_file(),'새 코드 push 필요'
check=subprocess.run([sys.executable,'-c','import numpy,pandas,matplotlib'],capture_output=True,text=True)
if check.returncode:
    subprocess.run([sys.executable,'-m','pip','install','-q','numpy','pandas','matplotlib'],check=True)
```

## Cell 2 — 변수

기본은 **세 baseline만** 분석한다. 다른 학습법을 분석하려면 해당 결과 폴더와 STRATEGY를 함께 바꾸고 새 OUTPUT_ROOT를 사용한다.

```python
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
MODELS=['xception','clip_b16','dinov2_b14']
STRATEGY='baseline'
PREDICTION_ROOT=ROOT/'evaluations/ffpp_c23_baselines_celebdf_official_smoothing_v1/seed42'
OUTPUT_ROOT=ROOT/'diagnostics/smoothing_logit_shift_v1'/f'{STRATEGY}_seed42'
MODEL_ARGS=' '.join(MODELS)
```

## Cell 3 — 분석

```python
!python -u "{PROJECT}/scripts/analyze_smoothing_logit_shift.py" --prediction-root "{PREDICTION_ROOT}" --output-root "{OUTPUT_ROOT}" --models {MODEL_ARGS} --strategy "{STRATEGY}"
assert _exit_code==0,'분석 실패: 위 오류 확인'
```

## Cell 4 — 표와 그림

기본 표는 변형이 실제 적용된 검출 성공 이미지 기준. SCOPE를 all로 바꾸면 fallback까지 포함한다.
중앙값은 유한 margin pair에서만 계산하며, 포화가 많으면 해당 subset이 전체를 대표한다고 단정하면 안 된다.

```python
import csv
from IPython.display import Markdown,Image,display
SCOPE='detected'  # all / detected / fallback
rows=list(csv.DictReader((OUTPUT_ROOT/'logit_shift_summary.csv').open()))
def number(value):
    return f'{float(value):.3f}' if value else 'N/A'
def percent(value):
    return f'{float(value):.2%}' if value else 'N/A'
table=['|모델|스무딩|종류|이미지 수|가짜 쪽 확률 이동|Δ로짓 중앙값|유한 pair 수|0/1 포화 비율|',
       '|---|---|---|---:|---:|---:|---:|---:|']
for r in rows:
    if r['scope']==SCOPE:
        table.append(f"|{r['model']}|{r['strength_percent']}%|{r['class']}|{r['images']}|{percent(r['probability_up_rate'])}|{number(r['delta_margin_median'])}|{r['finite_margin_pairs']}|{percent(r['saturation_rate'])}|")
display(Markdown('\n'.join(table)))
for model in MODELS:
    display(Image(filename=str(OUTPUT_ROOT/f'{model}_margin_shift.png')))
display(Image(filename=str(OUTPUT_ROOT/'probability_distributions.png')))
print('Saved:',OUTPUT_ROOT)
print('logit_shift_summary.csv: 요약 / *_paired_shift.csv: 같은 이미지의 전후 변화')
```
