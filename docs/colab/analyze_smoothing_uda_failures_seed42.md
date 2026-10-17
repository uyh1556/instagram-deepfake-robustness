# Smoothing_UDA_Error_Diagnosis_Seed42

Xception의 **Baseline → 일반 교사–학생 UDA → 판단 일치 선별 UDA**를 진단한다. 새 코드를 먼저 GitHub에 push하고 노트북을 실행한다. 재학습은 없다.

- **Cell 1–4: 저장된 테스트 예측 분석.** 이미지·TAR·GPU 없이 기존 CSV/JSON을 읽는다. 진짜의 오탐과 가짜의 미탐을 각각 얼마나 고쳤고 새로 만들었는지, 원본에서 맞혔다가 스무딩 때문에 틀린 이미지가 얼마나 구제됐는지 센다. 고정 raw margin 0(확률 기준 0.5), 기존 체크포인트를 그대로 사용하며 테스트로 임계값·에폭을 선택하지 않는다.
- **Cell 5–8: 선택적 GPU 임시 정답 진단.** 학습에 사용한 고정 비테스트 CDF 5,000장의 원본·약·중·강을 마지막에 저장된 두 EMA 교사로 추론한다. 기본 총 40,000 입력이다. 정답은 원본 비테스트 manifest에서 복원해 추론 후 정확도를 집계할 때만 사용한다. UDA를 다시 학습하거나 정답에 맞춰 데이터를 고르지 않는다.
- 마지막 `last.pt`에는 최종 EMA가 있으며 `best.pt`에는 검증으로 선택한 학생이 있다. **이 진단은 최종 EMA의 사후 평가다. 학습 당시 모든 step의 임시 정답이나 best 학생의 교사를 복원한 결과가 아니다.** 두 run의 종료 epoch도 다를 수 있다. 임시 정답 오류가 실제 학습 실패의 원인이라고 이 결과만으로 확정하지 않는다.
- CDF test 이미지와 적응 이미지를 섞지 않는다. 테스트 오류 분석과 타깃 임시 정답 진단의 출력도 분리한다. 동일 인물은 기존 정책대로 허용하며, 영상·이미지 중복 및 기존 보수적 영상 제외 목록을 확인한다.
- 타깃의 정답 비율은 바꾸지 않는다. 고정 pool의 이미지 평균과 실제 학습의 **영상균등 샘플링을 가정한 기대 가중 통계**를 구분한다. 임시 정답 채택 개수는 이번 고정·무반전 진단의 값이며 과거 실제 채택 개수가 아니다.
- Part B는 기존 non-test real/fake TAR 2개에서 **고정 타깃 이미지와 metadata만** 준비한다. FF++·테스트 TAR·새 ZIP·baseline 체크포인트는 필요 없다. 이미 준비된 타깃은 재사용한다. 최초 TAR 복사·스무딩 캐시는 시간이 들며, 진행률은 추가 추론에서 표시한다.
- 같은 이름의 Drive run 폴더가 중복될 수 있다. Cell 7은 코드 갱신 후 실행하고, 경로로 찾지 못하면 확인된 Xception 폴더 ID로 config/completed/last만 로컬 캐시에 읽는다. Google 인증이 한 번 표시될 수 있다. Drive 파일·폴더는 수정하지 않는다. 두 교사 위치를 먼저 확인한 뒤 추가 추론한다.
- 기본은 Xception이다. 다른 모델에서 `stable_agreement` 결과까지 완료한 경우에만 `MODELS`를 바꾼다. 추가 비교는 `ERROR_STRATEGIES`에 `random_agreement_count` 또는 `augmentation`을 넣을 수 있다. 임시 정답 진단은 일반 UDA와 판단 일치 UDA 두 개만 지원한다.

## Cell 1 — Drive·코드 준비

```python
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess, sys, json
PROJECT=Path('/content/instagram_deepfake_robustness')
if PROJECT.exists():
    subprocess.run(['git','-C',str(PROJECT),'pull','--ff-only'],check=True)
else:
    subprocess.run(['git','clone','https://github.com/uyh1556/instagram-deepfake-robustness.git',str(PROJECT)],check=True)
assert (PROJECT/'scripts/analyze_smoothing_uda_failures.py').is_file(),'새 코드를 먼저 GitHub에 push하세요.'
check=subprocess.run([sys.executable,'-c','import numpy,pandas,sklearn'],capture_output=True,text=True)
if check.returncode:
    subprocess.run([sys.executable,'-m','pip','install','-q','numpy','pandas','scikit-learn'],check=True)
print('저장 예측 분석 준비 완료. Cell 1–4는 CPU로 실행 가능합니다.')
```

## Cell 2 — 분석 대상·경로

```python
from datetime import datetime, timezone
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
MODELS=['xception']
ERROR_STRATEGIES=['mean_teacher','stable_agreement']
PSEUDO_STRATEGIES=['mean_teacher','stable_agreement']
EVALUATION_ROOTS=[
    ROOT/'evaluations/ffpp_c23_smoothing_mean_teacher_v1/seed42',
    ROOT/'evaluations/ffpp_c23_smoothing_stable_uda_v1/seed42',
]
HARD_MANIFEST_ROOT=ROOT/'manifests/ffpp_c23_smoothing_mean_teacher_v1/seed42'
HARD_RUNS=ROOT/'runs/ffpp_c23_smoothing_mean_teacher_v1'
STABLE_RUNS=ROOT/'runs/ffpp_c23_smoothing_stable_uda_v1'
CDF_DATA=ROOT/'data/celebdf_v2_face_crops16_v1'
TARGET_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
TARGET_TRAIN_CACHE=Path('/content/smoothing_mean_teacher_target_cache_v1')
STAMP=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
OUTPUT_ROOT=ROOT/'diagnostics/smoothing_uda_failure_v1'/f'seed42_{STAMP}'
ERROR_OUTPUT=OUTPUT_ROOT/'errors'
PSEUDO_OUTPUT=OUTPUT_ROOT/'pseudo'
MODEL_ARGS=' '.join(MODELS)
STRATEGY_ARGS=' '.join(ERROR_STRATEGIES)
PSEUDO_STRATEGY_ARGS=' '.join(PSEUDO_STRATEGIES)
ROOT_ARGS=' '.join(f'"{p}"' for p in EVALUATION_ROOTS)
PSEUDO_COMMON=f'--models {MODEL_ARGS} --strategies {PSEUDO_STRATEGY_ARGS} --seed 42 --hard-manifest-root "{HARD_MANIFEST_ROOT}" --analysis-root "{ERROR_OUTPUT}" --target-root "{TARGET_ROOT}" --output-root "{PSEUDO_OUTPUT}" --hard-runs-root "{HARD_RUNS}" --stable-runs-root "{STABLE_RUNS}" --target-train-cache "{TARGET_TRAIN_CACHE}"'
print('Models:',MODELS)
print('Output:',OUTPUT_ROOT)
```

## Cell 3 — 오류 변화 분석: 추가 추론 없음

```python
!python -u "{PROJECT}/scripts/analyze_smoothing_uda_failures.py" --evaluation-roots {ROOT_ARGS} --output-root "{ERROR_OUTPUT}" --models {MODEL_ARGS} --strategies {STRATEGY_ARGS}
assert _exit_code==0,'오류 분석 실패: 위 오류에서 평가 경로와 누락 조건 확인'
```

## Cell 4 — 오류 변화 결과 표시

`고친 오류 − 새 오류 = 순감소`다. 진짜의 오류는 FPR, 가짜의 오류는 FNR에 대응한다. `원본 정답→스무딩 오답` 집합은 **베이스라인을 기준**으로 고정한다. 후보가 해당 이미지를 고쳤더라도 후보의 원본 판단까지 유지했는지는 별도 열로 확인한다.

```python
import pandas as pd
from IPython.display import display, Markdown
labels={'baseline':'Baseline','mean_teacher':'일반 UDA','stable_agreement':'판단 일치 UDA',
        'random_agreement_count':'무작위 제외 UDA','augmentation':'Augmentation'}
conditions={'normal':'원본','smooth_weak':'약','smooth_medium':'중','smooth_strong':'강'}
means=pd.read_csv(ERROR_OUTPUT/'condition_means.csv')
transitions=pd.read_csv(ERROR_OUTPUT/'error_transitions.csv')
induced=pd.read_csv(ERROR_OUTPUT/'smoothing_induced_errors.csv')
for model in MODELS:
    display(Markdown(f'### {model} — 네 조건 평균'))
    t=means[means.model.eq(model)].copy()
    t['strategy']=t.strategy.map(labels)
    for col in ['fpr','fnr','ber']: t[col]=t[col].map(lambda x:f'{x:.2%}')
    t['auc']=t.auc.map(lambda x:f'{x:.4f}')
    display(t[['strategy','fpr','fnr','ber','auc']].rename(columns={'strategy':'방법'}))
    display(Markdown('### 베이스라인 대비 오류 변화'))
    t=transitions[transitions.model.eq(model)].copy()
    t['strategy']=t.strategy.map(labels); t['condition']=t.condition.map(conditions)
    t['class_name']=t.class_name.map({'real':'진짜: 오탐','fake':'가짜: 미탐'})
    t['error_reduction_pp']=t.error_reduction_pp.round(2)
    display(t[['strategy','condition','class_name','baseline_errors','repaired_errors','new_errors','candidate_errors','error_reduction_pp']].rename(columns={
        'strategy':'방법','condition':'조건','class_name':'오류 종류','baseline_errors':'기존 오류',
        'repaired_errors':'고친 오류','new_errors':'새 오류','candidate_errors':'학습 후 오류','error_reduction_pp':'오류 감소(%p)'}))
    display(Markdown('### 원본 정답 → 스무딩 오답 이미지의 구제 여부'))
    t=induced[induced.model.eq(model)&induced.condition.ne('normal')].copy()
    t['strategy']=t.strategy.map(labels); t['condition']=t.condition.map(conditions)
    t['class_name']=t.class_name.map({'real':'진짜','fake':'가짜'})
    display(t[['strategy','condition','class_name','baseline_smoothing_induced_errors','repaired','still_wrong','repaired_and_candidate_normal_correct','repaired_but_candidate_normal_wrong']].rename(columns={
        'strategy':'방법','condition':'조건','class_name':'실제 클래스','baseline_smoothing_induced_errors':'스무딩으로 생긴 오류',
        'repaired':'구제','still_wrong':'미해결','repaired_and_candidate_normal_correct':'구제+원본 정답 유지',
        'repaired_but_candidate_normal_wrong':'구제했으나 원본 오답'}))
print('이미지별 상세:',ERROR_OUTPUT/'paired_image_errors.csv')
print('점수 이동만으로 학습 실패 원인을 확정할 수는 없습니다.')
```

## Cell 5 — 선택적 Part B: GPU·추론 패키지

여기서 멈추면 Part A만 완료된다. Part B까지 진행할 때는 GPU 런타임을 사용한다. 이미 GPU에서 실행 중이면 그대로 이어간다. 패키지는 별도 Python 프로세스에서 확인하므로 노트북에 로드된 Pillow를 검사해 강제로 중단시키지 않는다.

```python
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(),"Part B는 GPU 런타임에서 실행하세요."; print(torch.cuda.get_device_name(0))'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','from PIL import Image, ImageText; import cv2; print("Pillow:",Image.__version__,"OpenCV:",cv2.__version__)'],check=True)
```

## Cell 6 — 고정 적응 이미지 준비: 테스트 TAR 불필요

```python
!python -u "{PROJECT}/scripts/diagnose_smoothing_pseudo_labels.py" prepare {PSEUDO_COMMON} --drive-data "{CDF_DATA}"
assert _exit_code==0,'고정 타깃 준비 실패: 위 오류 확인. 기존 파일은 삭제하지 마세요.'
```

## Cell 7 — 마지막 EMA 교사의 임시 정답 진단

새 학습은 없다. 두 최종 EMA를 각각 한 번 로딩하고, 고정 pool을 네 강도로 추론한다. 진행률은 이미지 묶음 단위다. 완성된 교사별 CSV/JSON은 같은 입력·코드·체크포인트 해시일 때 재사용한다. 이 셀만 재실행할 때 Cell 2를 다시 실행하지 않으면 같은 출력 경로를 재사용한다.

```python
subprocess.run(['git','-C',str(PROJECT),'pull','--ff-only'],check=True)
import shlex
DRIVE_RUN_IDS={
    'xception/mean_teacher':'1WNxM9vAh02RBXIY7mzjOxSe_QYMaEFP-',
    'xception/stable_agreement':'1304Z4iSJpDUF9PVw3goELzpl7JdjKinL',
}
expected=[(HARD_RUNS if strategy=='mean_teacher' else STABLE_RUNS)/f'{model}_{strategy}_ffpp_c23_smoothing_{"mean_teacher" if strategy=="mean_teacher" else "stable_uda"}_v1_seed42'/'config.json'
          for model in MODELS for strategy in PSEUDO_STRATEGIES]
if any(not path.is_file() for path in expected):
    try:
        import google.auth, googleapiclient.discovery
    except ImportError:
        subprocess.run([sys.executable,'-m','pip','install','-q','google-api-python-client','google-auth'],check=True)
    from google.colab import auth
    auth.authenticate_user()
DRIVE_ID_ARG=shlex.quote(json.dumps(DRIVE_RUN_IDS))
!python -u "{PROJECT}/scripts/diagnose_smoothing_pseudo_labels.py" evaluate {PSEUDO_COMMON} --drive-run-ids {DRIVE_ID_ARG} --batch-size 32 --workers 2
assert _exit_code==0,'임시 정답 진단 실패: 위 오류 확인 후 Cell 7 재실행'
```

## Cell 8 — 임시 정답 정확도·오류 방향 표시

`confidence_selected`는 확신도 조건만 적용한 경우, `agreement_selected`는 같은 교사에서 판단 일치까지 요구한 경우다. `protocol_selected`는 각 교사의 실제 프로토콜 규칙을 현재 snapshot에 적용한 결과다. 같은 교사에서 일치 조건을 더했을 때 정확도와 채택량이 어떻게 바뀌는지 확인할 수 있다. 정답 비율을 5:5로 조정한 결과가 아니다.

```python
quality=pd.read_csv(PSEUDO_OUTPUT/'pseudo_quality.csv')
rules={'confidence_selected':'확신도 ≥0.9','agreement_selected':'확신도+네 강도 판단 일치','protocol_selected':'해당 UDA 규칙'}
for model in MODELS:
    display(Markdown(f'### {model} — 최종 EMA 임시 정답 품질'))
    t=quality[quality.model.eq(model)&quality.grouping.eq('all')&quality.rule.eq('protocol_selected')].copy()
    t['strategy']=t.strategy.map(labels); t['condition']=t.condition.map(conditions)
    for col in ['selected_accuracy','video_uniform_selected_accuracy']:
        t[col]=t[col].map(lambda x:'없음' if pd.isna(x) else f'{x:.2%}')
    display(t[['strategy','condition','selected','selected_wrong_real_as_fake','selected_wrong_fake_as_real','selected_accuracy','video_uniform_selected_accuracy']].rename(columns={
        'strategy':'교사','condition':'조건','selected':'채택 이미지','selected_wrong_real_as_fake':'진짜→가짜 오답',
        'selected_wrong_fake_as_real':'가짜→진짜 오답','selected_accuracy':'채택 정답률: 이미지 평균',
        'video_uniform_selected_accuracy':'채택 정답률: 영상균등 기대 가중'}))
    display(Markdown('### 같은 교사에서 판단 일치 조건을 더했을 때'))
    t=quality[quality.model.eq(model)&quality.grouping.eq('all')&quality.rule.ne('protocol_selected')].copy()
    t['strategy']=t.strategy.map(labels); t['condition']=t.condition.map(conditions); t['rule']=t.rule.map(rules)
    display(t[['strategy','condition','rule','selected','selected_correct','selected_wrong','selected_accuracy']].rename(columns={
        'strategy':'교사','condition':'조건','rule':'선별 기준','selected':'채택','selected_correct':'정답','selected_wrong':'오답','selected_accuracy':'정답률'}))
print('최종 EMA의 사후 진단입니다. 학습 당시 모든 step의 임시 정답이나 오류의 인과관계를 복원한 결과는 아닙니다.')
print('임시 정답 상세:',PSEUDO_OUTPUT)
print('전체 진단:',OUTPUT_ROOT)
```

주요 출력:

- `errors/error_transitions.csv`: 클래스별 고친 오류·새 오류·순변화.
- `errors/smoothing_induced_errors.csv`: 베이스라인에서 스무딩으로 생긴 오류의 구제와 후보 원본 성능 유지 여부.
- `errors/paired_image_errors.csv`: 같은 이미지의 기존/후보 margin·확률·정오답.
- `errors/smoothing_flips.csv`, `margin_changes.csv`: 각 모델의 스무딩 판정 변화 및 후보 점수 이동.
- `pseudo/pseudo_quality.csv`: 클래스·예측 클래스·선별 규칙별 임시 정답 정확도.
- `pseudo/xception_*.csv`: 고정 비테스트 이미지의 네 강도 최종 교사 예측. 실제 정답은 진단용이다.
- 두 protocol JSON: 입력/체크포인트 해시, 마지막 교사 epoch, 선택 학생 epoch, 해석 한계.
