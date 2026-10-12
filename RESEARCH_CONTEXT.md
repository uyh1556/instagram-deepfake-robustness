# Research Context

## 현재 연구 방향
FF++ raw/c0로 학습한 Xception에 Instagram 색감 필터를 적용했을 때의 변화와 augmentation/Domain Adaptation을 탐색한다. 현재는 파일 준비 단계이며, seen/unseen 필터 구성·DA 방법·학습 프로토콜은 확정하지 않았다.

## 데이터와 고정 분할
- 기존 프로젝트의 전체 프레임 PNG 27,529장을 복사했다. 원본 영상 전체 또는 FF++ 전체 데이터셋이라고 표현하지 않는다.
- Original → real, Deepfakes·Face2Face → fake. 조작 기법 metadata는 유지한다.
- Custom source-aware split seed42: train 19,267장, val 4,130장, test 4,132장.
- 공식 split이 아니다. 기존 source 연결 관계에 기반한 분할을 그대로 유지했다.
- 기본 manifest SHA256: 3deca3de17f28e9be112e9e6f884fecaaf99fddccaa88355ce3737ef4ea13496
- Face ROI manifest SHA256: 00c5d1e97b245a73a64aa2446595a37153d66d6dfe113cb82df1076a44a5e162
- 경로 root는 `data/ffpp_c0_v1`; CSV의 `source_images_raw/` prefix를 보존했다.

## 재사용 모델
Standard, 전체 프레임 Letterbox299, 고정 얼굴 ROI 1.5배의 seed42 best.pt와 설정·학습 기록을 복사했다. 실제 state_dict 로딩과 checkpoint 내부 manifest 해시를 검증했다. 모델별 기존 crop/resize/normalize를 사용하고, DF40의 canonical256 + Q95 전처리를 자동으로 적용하지 않는다.
원본 config.json의 이전 경로는 출처 기록이므로 수정하지 않는다. 실행 경로는 configs/assets.json과 명시적인 CLI 인자로 설정한다.

## 필터
사용자 제공 61개 항목에는 Normal이 포함된다. configs/filters의 원문을 보존했다. Lo-Fi는 사용자 확인 완료, Willo/Willow 표기는 미확인이다. pilgram2는 Instagram-like 근사 구현으로, 실제 앱과 동일한 처리라고 주장하지 않는다. 필터 적용 데이터는 아직 생성하지 않았다.

## 준비와 검증
configs/assets.json에 출처·경로·Drive ID·이미지 수·manifest 및 checkpoint 해시가 있다. 이미지 경로와 크기, CSV 바이트, ROI와 기본 manifest의 샘플/분할 일치, 모델 로딩 및 TAR 구성원을 확인했다. 모든 이미지의 내용 해시를 전수 비교한 것은 아니다.
Drive에는 모델·manifest·기존 원본 평가 기록을 복사했다. TAR 3개는 사용자 직접 업로드 대상으로 안내했다. 추후 완료 여부는 Drive 파일 목록에서 확인한다.
새 성능 평가나 학습은 수행하지 않았다. outputs/reference는 이전 실험의 결과이며 새 연구 결과로 소개하지 않는다.

## 다음 단계
사용자 TAR 업로드 확인 → 동일 원본 데이터로 baseline 재현 → 필터 파일럿 설계. 학습/평가용 source 분할을 유지하며, augmentation 비교와 DA의 target 데이터 사용 범위를 실험 전에 명확히 한다.
