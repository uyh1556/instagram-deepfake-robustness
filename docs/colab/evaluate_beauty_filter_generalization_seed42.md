# Beauty_Filter_Generalization_Seed42 (모델 공용: Xception / CLIP / DINO, 추론만)

**질문:** bilateral 짝으로만 학습한 ablation 모델들이 **학습에 쓰지 않은 피부 스무딩 필터**에도 강건한가?
그리고 그 일반화에서도 순위 손실(`aug_pair_rank`)이 증강(`aug_single`, `aug_pair`)보다 나은가?
Cell 2의 `MODELS`에 모델 **하나**를 넣는다. 모델마다 이 노트북을 따로 열어 동시에 실행해도 된다.

- 평가 모델: Baseline, `clean_only`, `aug_single`, `aug_pair`, `aug_pair_rank` (source ablation seed42의 best.pt). 학습·선택 없음.
- 평가 필터: weighted median, fast global smoother, domain transform — 보정값(`manifests/beauty_filter_calibration_v1/calibration.json`, FF++ train으로 bilateral과 피부 변화량을 맞춘 값), 같은 피부 마스크, 약·중·강 = 0.25/0.5/0.75.
- 원본과 bilateral 예측은 ablation 평가에서 **재사용**한다(checkpoint 해시 확인). 따라서 **ablation Cell 6 평가가 끝나 있어야 한다.**
- 지표(필터마다 원본+약·중·강): **FP gap**(`fp_gap`, 보정된 진짜 = 오탐 방향)과 **FN gap**(`fn_gap`, 보정된 가짜 = 미탐 방향), 쏠림, FP·FN 뒤집힘, 점수 이동. 영상 부트스트랩으로 각 조건 − `aug_pair_rank`의 두 방향 격차 차이.
- **캐시:** 필터 결과(섞기 전 100%)를 이미지당 한 번만 계산해 `/content`에 두고 다섯 checkpoint·세 강도가 함께 쓴다.
  처음 끝낸 노트북이 Drive에 tar로 올려 두면(`manifests/beauty_filter_calibration_v1/cdf_test_filter_cache_v1.tar`) 다른 노트북은 그걸 풀어 쓴다.
  세 노트북을 동시에 시작하면 각자 계산할 수 있다(결과는 같고 tar는 먼저 끝난 것 하나만 남는다).
- 예측은 Drive `evaluations/beauty_filter_generalization_<모델>/seed42/`에 저장되어 끊겨도 다시 실행하면 이어서 한다.

## Cell 1 — 설치·코드 갱신 (OpenCV contrib 포함)

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
assert (PROJECT/'scripts/evaluate_beauty_filter_generalization.py').is_file(),'새 코드를 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
# Same OpenCV version as the pinned headless build, plus contrib (ximgproc); bilateral stays identical.
subprocess.run([sys.executable,'-m','pip','install','-q','opencv-contrib-python-headless==4.11.0.86'],check=True)
subprocess.run([sys.executable,'-c','import cv2,torch; assert hasattr(cv2,"ximgproc"),"contrib 미설치"; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(cv2.__version__, torch.cuda.get_device_name(0))'],check=True)
```

## Cell 2 — 모델·경로·중복 폴더 검사

```python
MODELS=['clip_b16']  # 노트북마다 하나: ['xception'] / ['clip_b16'] / ['dinov2_b14']
assert len(MODELS)==1,'동시 실행 시 노트북마다 모델 하나'
GROUP=MODELS[0]
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
CDF_DATA=ROOT/'data/celebdf_v2_face_crops16_v1'
TARGET_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
CALIBRATION_DIR=ROOT/'manifests/beauty_filter_calibration_v1'
ABLATION=f'ffpp_c23_smoothing_source_ablation_{GROUP}'
EXPERIMENT='beauty_filter_generalization'
OUTPUT=ROOT/'evaluations'/f'{EXPERIMENT}_{GROUP}'/'seed42'
COMMON=(f'--models {GROUP} --seed 42 --target-root "{TARGET_ROOT}" --baseline-runs-root "{ROOT}/runs/ffpp_c23_4methods_baseline_v1" '
        f'--runs-root "{ROOT}/runs/{ABLATION}" --ablation-output-root "{ROOT}/evaluations/{ABLATION}/seed42" '
        f'--calibration "{CALIBRATION_DIR}/calibration.json" --output-root "{OUTPUT}" '
        f'--filter-cache /content/beauty_filter_cache_v1 --cache-archive "{CALIBRATION_DIR}/cdf_test_filter_cache_v1.tar"')
assert (CALIBRATION_DIR/'calibration.json').is_file(),'필터 보정(calibrate_beauty_filters)을 먼저 실행하세요.'
assert (ROOT/'evaluations'/ABLATION/'seed42'/'summary.csv').is_file(),'source ablation Cell 6 평가를 먼저 끝내세요.'
print('Model:',GROUP,'| Results:',OUTPUT,flush=True)

try:
    import googleapiclient.discovery
except ImportError:
    subprocess.run([sys.executable,'-m','pip','install','-q','google-api-python-client','google-auth'],check=True)
from google.colab import auth
auth.authenticate_user()
sys.path.insert(0,str(PROJECT/'scripts'))
import drive_folder_guard
drive_folder_guard.check([OUTPUT],model_tag=GROUP)
```

## Cell 3 — CDF 공식 test 준비

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

## Cell 4 — 필터 캐시 (한 번만, CPU 병렬)

```python
!python -u "{PROJECT}/scripts/evaluate_beauty_filter_generalization.py" prepare {COMMON} --cache-workers {os.cpu_count()}
assert _exit_code==0,'캐시 준비 실패: 위 오류 확인 후 다시 실행하면 이어서 계산'
```

## Cell 5 — 평가

```python
!python -u "{PROJECT}/scripts/evaluate_beauty_filter_generalization.py" evaluate {COMMON} --eval-batch-size 64 --workers 2 --bootstrap 1000
assert _exit_code==0,'평가 실패: 위 오류 확인 후 Cell 5 재실행(끝난 예측은 재사용)'
```

## Cell 6 — 결과 표시

```python
import csv
from IPython.display import Markdown,display
names={'baseline':'Baseline','clean_only':'① 원본만','aug_single':'② 증강(한 장)','aug_pair':'③ 증강(짝)','aug_pair_rank':'④ 증강(짝)+순위'}
filters=['bilateral','weighted_median','fast_global','domain_transform']
rows=list(csv.DictReader((OUTPUT/'summary.csv').open()))
cell={(r['strategy'],r['filter']):r for r in rows}
for metric,title,fmt in [('fp_gap','FP gap (보정된 진짜, 오탐 방향) ↓','.4f'),('fn_gap','FN gap (보정된 가짜, 미탐 방향)','.4f'),
                         ('fp_cross_auc','강 진짜 vs 원본 가짜 AUC ↑','.4f'),('fp_flip@normal5','FP 뒤집힘 ↓','.2%'),('fn_flip@normal5','FN 뒤집힘 ↓','.2%')]:
    lines=[f'### {GROUP} — {title} (bilateral = 학습 필터, 나머지 = 처음 보는 필터)','','|조건|'+'|'.join(filters)+'|','|---|'+'---:|'*len(filters)]
    for s,label in names.items():
        lines.append(f'|{label}|'+'|'.join(format(float(cell[(s,f)][metric]),fmt) for f in filters)+'|')
    display(Markdown('\n'.join(lines)))
boot=list(csv.DictReader((OUTPUT/'bootstrap_gaps_vs_aug_pair_rank.csv').open()))
side={'fp_gap':'FP gap (보정된 진짜)','fn_gap':'FN gap (보정된 가짜)'}
lines=['### 격차 차이 (조건 − ④), 영상 부트스트랩 95% 구간','','|필터|방향|조건|차이|95% 구간|','|---|---|---|---:|---|']
for r in boot:
    lines.append(f"|{r['filter']}|{side[r['metric']]}|{names[r['strategy']]}|{float(r['observed']):+.4f}|[{float(r['ci_low']):+.4f}, {float(r['ci_high']):+.4f}]|")
display(Markdown('\n'.join(lines)))
print('양수이고 구간이 0보다 크면 ④가 그 조건보다 그 방향 격차가 작다는 뜻입니다. 불변이려면 두 방향 모두 0 근처여야 합니다. 원본 AUC는 필터와 무관하게 같습니다.')
print('Results:',OUTPUT)
```
