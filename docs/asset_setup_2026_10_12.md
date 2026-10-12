# FF++ c0 준비 기록 — 2026-10-12

- 이미지: 27,529장; original 9,181 / Deepfakes 9,158 / Face2Face 9,190.
- split: 학습 19,267 / 검증 4,130 / 테스트 4,132. 기존 CSV를 그대로 복사함.
- APFS 독립 복사로 이미지 파일을 준비함. symlink로 연결한 데이터가 아님.
- Standard·Letterbox·Face ROI seed42 best.pt, config·학습 기록·기존 원본 평가 결과 복사.
- 체크포인트의 state_dict와 manifest SHA256 일치 확인.
- 각 TAR의 이미지 경로·개수·파일 크기가 로컬 이미지와 일치함.
- 새 필터 평가나 학습은 수행하지 않음.

## Drive
프로젝트: https://drive.google.com/drive/folders/1BDyboyTlm4aKiJ0LLJDkRknAfWXZZwxx
데이터 TAR 업로드 위치: https://drive.google.com/drive/folders/1rfSWqk2dk6b2MDVJgl-9wLbnaj_x5V8u

연결 도구의 파일당 512MiB 제한으로 TAR 업로드는 사용자 직접 진행. 모델·manifest·기존 평가 결과는 복사 완료.

## TAR
| 파일 | 이미지 수 | 크기 |
|---|---:|---:|
| ffpp_c0_original_v1.tar | 9181 | 6.12 GiB |
| ffpp_c0_deepfakes_v1.tar | 9158 | 6.11 GiB |
| ffpp_c0_face2face_v1.tar | 9190 | 6.13 GiB |

로컬 위치: `data/exports/`. 압축 해제 위치는 `ffpp_c0_v1/source_images_raw/<method>/`.

필터 목록은 루트의 Instagram_filters.txt에서 configs/filters/instagram_app_filters_2026_10_12.txt로 이동했다. 원문은 변경하지 않았다.
기존 프로젝트와 Drive 원본 파일은 변경·삭제하지 않았다.
