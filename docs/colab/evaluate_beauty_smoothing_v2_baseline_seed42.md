# Beauty_Smoothing_V2_Baseline_Seed42 (모델 공용: Xception / CLIP / DINO, 추론만)

**질문:** 새 v2 스무딩(파싱 피부 마스크 + 경계 보존 필터 4개, 약·중·강 25/50/75%)에서도 FF++ Baseline의 점수가 스무딩 때문에 쏠리는가?
이 결과에서 쏠림이 확인돼야 대각선 학습(필터 k로 학습 → k로 평가)으로 넘어간다.

- Cell 2의 `MODELS`에 모델 **하나**를 넣는다. 모델마다 이 노트북을 따로 열어 동시에 실행해도 된다.
- `DATASETS`: `cdf_test`(CDF 공식 test, 다른 도메인), `ffpp_test`(FF++ test, 같은 도메인: CDF 도메인 차이 없이 스무딩만의 영향), `cdf_dev`(CDF 비테스트 개발 세트: 이후 새 학습법을 비교할 때의 Baseline 기준값).
- 먼저 `build_beauty_smoothing_v2_cache`로 해당 데이터셋의 tar 5개(mask + 필터 4개)를 만들어야 한다. 여기서는 풀기만 한다.
- 지표(필터마다): 원본 AUC, **FP gap**(보정된 진짜 = 오탐 방향), **FN gap**(보정된 가짜 = 미탐 방향), max|gap|, 교차 AUC, 뒤집힘, 점수 이동. 테스트 라벨은 채점에만 쓴다.
- 예측은 Drive `evaluations/beauty_smoothing_v2_baseline_<모델>/seed42/<데이터셋>/`에 저장되어 끊겨도 이어서 한다.

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
assert (PROJECT/'scripts/evaluate_beauty_smoothing_v2.py').is_file(),'새 코드를 GitHub에 push하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','setuptools==80.9.0','wheel'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation','-r',str(PROJECT/'requirements-ffpp-c23-smoothing-eval.txt')],check=True)
subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(),"GPU 런타임 선택"; print(torch.cuda.get_device_name(0))'],check=True)
```

## Cell 2 — 모델·경로

```python
MODELS=['clip_b16']  # 노트북마다 하나: ['xception'] / ['clip_b16'] / ['dinov2_b14']
DATASETS=['cdf_test','ffpp_test','cdf_dev']
assert len(MODELS)==1,'동시 실행 시 노트북마다 모델 하나'
GROUP=MODELS[0]
ROOT=Path('/content/drive/MyDrive/instagram-deepfake-robustness')
ARCHIVE=ROOT/'manifests/beauty_smoothing_v2/cache'
CDF_ROOT=Path('/content/celebdf_v2_face_crops16_v1')
FF_ROOT=Path('/content/ffpp_c23_native_roi16_v1')
DATA_ROOT={'cdf_test':CDF_ROOT,'ffpp_test':FF_ROOT,'cdf_dev':CDF_ROOT}
MEMBERSHIP=next((ROOT/'manifests/ffpp_c23_smoothing_rank_sda_pool_v2/seed42'/m/'pool_v2/membership_labelled.csv' for m in ['clip_b16','dinov2_b14','xception']
                 if (ROOT/'manifests/ffpp_c23_smoothing_rank_sda_pool_v2/seed42'/m/'pool_v2/membership_labelled.csv').is_file()),None)
assert 'cdf_dev' not in DATASETS or MEMBERSHIP is not None,'pool v2 membership_labelled.csv를 찾지 못했습니다'
OUTPUT=ROOT/'evaluations'/f'beauty_smoothing_v2_baseline_{GROUP}'/'seed42'
for d in DATASETS:
    for part in ['mask','bilateral','weighted_median','fast_global','domain_transform']:
        assert (ARCHIVE/f'{d}_{part}.tar').is_file(),f'캐시 노트북을 먼저 실행하세요: {d}_{part}.tar'
print('Model:',GROUP,'| Results:',OUTPUT,flush=True)

# 결과 폴더의 첫 새 단계가 모델별(<실험>_<모델>)이라 다른 모델 노트북과 동시에 실행해도 같은 폴더를 만들지 않는다.
# 그래서 Drive API 중복 폴더 검사(두 번째 구글 인증)는 생략한다. 같은 모델을 두 노트북에서 동시에 돌리지만 않으면 된다.
```

## Cell 3 — 테스트 데이터 준비

```python
def extract_cdf(name,size):
    if (CDF_ROOT/'manifests'/f'{name.removesuffix(".tar")}.csv').is_file(): return
    source=ROOT/'data/celebdf_v2_face_crops16_v1'/name
    assert source.is_file() and source.stat().st_size==size,f'TAR 확인: {source}'
    assert shutil.disk_usage('/content').free>2*size+2*2**30,'Colab 로컬 공간 부족'
    local=Path('/content')/name; print('Copying:',name,flush=True); shutil.copy2(source,local)
    subprocess.run(['tar','-xf',str(local),'-C','/content'],check=True); local.unlink()
if {'cdf_test','cdf_dev'} & set(DATASETS):
    extract_cdf('celebdf_v2_roi16_test_v1.tar',557015040)
if 'cdf_dev' in DATASETS:
    extract_cdf('celebdf_v2_roi16_non_test_real_v1.tar',826142720)
    extract_cdf('celebdf_v2_roi16_non_test_fake_v1.tar',5679564800)
if 'ffpp_test' in DATASETS:
    !python -u "{PROJECT}/scripts/prepare_ffpp_c23_training_data.py" --drive-data "{ROOT}/data/ffpp_c23_4methods_roi16_v1" --data-root "{FF_ROOT}" --test-only
    assert _exit_code==0,'FF++ test 준비 실패'
```

## Cell 4 — 평가 (끊기면 다시 실행: 끝난 예측은 재사용)

```python
for d in DATASETS:
    extra=f'--membership "{MEMBERSHIP}"' if d=='cdf_dev' else ''
    !python -u "{PROJECT}/scripts/evaluate_beauty_smoothing_v2.py" --dataset {d} {extra} --data-root "{DATA_ROOT[d]}" --baseline-runs-root "{ROOT}/runs/ffpp_c23_4methods_baseline_v1" --archive-dir "{ARCHIVE}" --output-root "{OUTPUT}/{d}" --models {GROUP} --eval-batch-size 64 --workers 2
    assert _exit_code==0,f'{d} 평가 실패: 위 오류 확인 후 다시 실행'
```

## Cell 5 — 결과 표시

```python
import csv
from IPython.display import Markdown,display
filters=['bilateral','weighted_median','fast_global','domain_transform']
for d in DATASETS:
    rows={r['filter']:r for r in csv.DictReader((OUTPUT/d/'summary.csv').open())}
    lines=[f'### {GROUP} Baseline — {d} (v2 스무딩, 강 0.75 기준)','','|필터|원본 AUC|FP gap|FN gap|max\\|gap\\||FP 뒤집힘|FN 뒤집힘|진짜 이동σ|가짜 이동σ|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for f in filters:
        r=rows[f]
        lines.append(f"|{f}|{float(r['normal_auc']):.4f}|{float(r['fp_gap']):+.4f}|{float(r['fn_gap']):+.4f}|{float(r['max_gap']):.4f}|"
                     f"{float(r['fp_flip@normal5']):.2%}|{float(r['fn_flip@normal5']):.2%}|{float(r['real_shift']):+.3f}|{float(r['fake_shift']):+.3f}|")
    display(Markdown('\n'.join(lines)))
old=ROOT/'evaluations'/f'ffpp_c23_smoothing_source_ablation_{GROUP}'/'seed42'/'summary.csv'
if old.is_file():
    r=[x for x in csv.DictReader(old.open()) if x['strategy']=='baseline'][0]
    print(f"참고 — 이전 v1(기하 마스크 bilateral, 약 2/255) CDF Baseline: FP gap {float(r['fp_gap']):+.4f}, FN gap {float(r['fn_gap']):+.4f}")
print('FP gap 양수 = 보정된 진짜가 가짜처럼 보임, FN gap 음수 = 보정된 가짜가 더 가짜처럼 보임(둘 다 가짜 쪽 쏠림). 0 근처면 쏠림 없음.')
print('Results:',OUTPUT)
```
