# Jetson 카메라 응용 실습 8~10단계 (AIGJ_01_010 박수연)

Jetson Orin Nano와 USB 웹캠으로 진행한 EmbeddingGemma 2 실습의 카메라 응용 단계입니다. 웹캠 물체 추출과 텍스트 검색(8단계), 사람 동작·행동 인식(9단계), 카메라 2대 근접 알람(10단계)을 담았습니다.

- 1~7단계(환경 구축, 텍스트·이미지·영상 임베딩, 성능 측정)는 [embedding_gemma2-AIGJ_01_010](https://github.com/AIP-1/embedding_gemma2-AIGJ_01_010) 저장소에 있습니다.
- 실습 안내문은 [docs/guide_README.md](docs/guide_README.md)에 있습니다.
- 공유 보드 B에서 팀원(AIGJ_01_004)과 함께 진행했습니다. 코드와 측정 기록은 같은 실험에서 나온 것이며, 팀 저장소는 [practice_task_for_jetson_camera_applications-AIGJ_01_004](https://github.com/AIP-1/practice_task_for_jetson_camera_applications-AIGJ_01_004)입니다.

## 환경 (보드 B, team5-b)

| 항목 | 값 |
| --- | --- |
| 보드 | Jetson Orin Nano 8GB, L4T R39 (JetPack 7 계열), MAXN_SUPER (nvpmodel 2번), 스왑 8GB |
| 파이썬 | Python 3.12, venv `~/eg2` |
| 라이브러리 | torch 2.14.1+cu132, sentence-transformers 6.2.0, ultralytics 8.4.184, OpenCV |
| 모델 | google/embeddinggemma-2 (bfloat16), yolo11n, yoloe-11s-seg, yolo11n-pose |
| 카메라 | Logitech C920 2대 (/dev/video0, /dev/video2), 자동초점 끔 |

## 진행 단계

| 단계 | 내용 | 상태 |
| --- | --- | --- |
| 8 | 웹캠 물체 추출과 웹 검색 (YOLO, 임베딩 재라벨, YOLOE 비교) | 완료 |
| 9 | 사람 동작 인식(만세·한팔들기·손 흔들기·악수), 행동 패턴(관심도), GRU 학습 | 완료 |
| 10 | 두 카메라 보정, ankle1·stereo·fused 거리 비교, 타겟 자동 찾기, 이동 타겟 충돌 알람 | 완료 |

## 구조

```
USB 웹캠 → Jetson
  ├ 캡처 스레드: 프레임 → YOLO 계열 검출·추적 → 크롭 / 관절
  ├ 임베딩 스레드: EmbeddingGemma 2 → 메모리 인덱스 (512차원, 최대 1000개)
  └ FastAPI 웹 서버 (:8000) → 브라우저 (실시간 영상, 텍스트 검색, 알람 기록)
```

영상과 임베딩을 다른 스레드로 나누고 사이에 큐를 둬서, 임베딩이 느려도 영상은 15 fps로 계속 흐르게 했습니다. 사람 이미지는 크롭·저장하지 않고 좌표와 알람 기록만 남깁니다.

## 폴더 구성

```
app.py              8단계 물체 추출 + 웹 검색 서버 (--mode yolo|relabel|yoloe)
pose_app.py         9단계 동작·행동 인식 (--raw, --record, --behavior, --model)
pose_model.py       9단계 관절 특징(53개)과 GRU 모델
train_pose.py       9단계 학습 스크립트
data/action/        9단계 동작 학습 데이터 (관절 좌표 csv)
data/behavior/      9단계 행동 학습 데이터 (관절 좌표 csv)
prox/               10단계 두 카메라 근접 알람
  calib_intrinsics.py   렌즈 보정 (체커보드)
  calib_extrinsics.py   카메라 위치 보정 (기준점 클릭, solvePnP)
  stereo.py             삼각측량, 한 카메라 추정, 칼만 필터
  proximity_app.py      근접·이탈·충돌 알람 앱
  calib/                기준점 좌표, 클릭 좌표, 측정 기록
  check/                카메라 위치 변화 확인 이미지
results/            8, 9, 10단계 측정 결과
reports/            PDF 보고서 (9단계, 10단계, 1~10단계 전체)
docs/               실습 안내문
```

## 실행

```bash
source ~/eg2/bin/activate && cd ~/eg2-lab
python app.py --mode relabel                      # 8단계
python pose_app.py --behavior                     # 9단계 (규칙)
python pose_app.py --model models/action.pt       # 9단계 (학습 모델)
cd prox && python proximity_app.py --cams 0 2 --target "brown cardboard box"            # 10단계
python proximity_app.py --cams 0 2 --target "brown cardboard box" --moving --ttc 3.0     # 10-8 이동 타겟
# 브라우저: http://<Jetson IP>:8000
```

## 주요 결과

### 8단계 물체 추출: 세 방식 비교

| 물체 | yolo | relabel | yoloe |
| --- | --- | --- | --- |
| 무선 이어폰 | remote / mouse ✗ | 무선 이어폰 ✓ | mouse ✗ |
| 드럼 세탁기 | microwave ✗ | 드럼 세탁기 ✓ | drum washing machine ✓ |
| 선풍기 | clock ✗ | 천장 에어컨 ✗ (목록 추가 전) | electric fan ✓ |
| 시계 | clock ✓ | 휴대폰 ✗ (목록 추가 전) | wall clock ✓ |
| **검출 fps** | 15 | 15~17 | 13 |

- YOLO(COCO 80종)는 모르는 물체도 80종 중 하나로 부른다(닫힌 어휘).
- 임베딩 재라벨은 학습 없이 목록만으로 바로잡지만, 목록 밖 물체는 오히려 틀린다.
- YOLOE는 글로 클래스를 정해 COCO 밖 물체도 찾지만, 작고 비슷한 물체는 헷갈린다.
- **선택: YOLOE로 후보를 찾고 헷갈리는 물체는 임베딩으로 확인하는 조합.** 10단계 타겟 찾기가 이 방식이다.

### 9단계 동작·행동 인식

- 프레임 하나로는 손 흔들기를 판정할 수 없어, 0.6초 다수결과 1.5초 방향 전환 판정으로 해결했다.
- 동작 GRU(window 30, idle·wave·clap·arms_crossed) 검증 정확도 0.989.
- 실시간에서 "가만히 있어도 wave" 문제가 있었다. 얼굴을 빼고 녹화해 얼굴 특징이 항상 0이었고, 표준화에서 표준편차가 0에 가까워 값이 폭주한 것이 원인이었다. 표준편차 최소값을 0.03으로 두고 박수 데이터를 더 녹화해 4개 동작 모두 실시간 인식.
- 행동 패턴(지나감·힐끗 봄·관심 있음·머무름) 규칙은 4개 시나리오 × 3회 모두 기대한 결과. 행동 모델은 한 번 지나가는 시간이 짧아 window 15로 학습했다.

### 10단계 카메라 2대 근접 알람

- 렌즈 보정 RMS 0.372 / 0.355 px. 위치 보정 재투영 오차 3.78 / 3.22 px, 검증점 오차 3.1 / 3.0 cm.

| 실제 거리·자세 | ankle1 (카메라 1대, 발목) | stereo (2대, 어깨 삼각측량) | fused |
| --- | --- | --- | --- |
| 0.5 m 서 있음 | 0.56 | 0.61 | 0.63 |
| 1.0 m 서 있음 | 1.18 | 1.06 | 1.21 |
| 2.0 m 서 있음 | 2.05 | 2.14 | 2.11 |
| 0.81 m 책상에 앉음 | 1.45 (ID 끊김) | **0.67** | 0.91 |
| fps | 26.7 | 23.1 | 23 |

- 앉으면 책상에 발목이 가려져 카메라 1대 방식이 실패했고, 책상 위로 보이는 어깨를 두 카메라로 삼각측량해 해결했다.
- 알람은 히스테리시스(0.2 m)로 경계에서 깜빡이지 않게 했다. 타겟 자동 찾기 후 알람 시나리오 5개 모두 3/3 성공.
- 이동 타겟은 칼만 필터로 속도를 추정하고 TTC(충돌 예상 시간)로 알람을 냈다. 충돌 위험 알람이 근접 알람보다 천천히 걸을 때 약 3초, 보통 속도일 때 약 1초 먼저 울렸다.

자세한 측정값은 [results/](results/)와 [reports/](reports/)에 있습니다.

## 트러블슈팅

| 단계 | 문제 | 원인 | 해결 |
| --- | --- | --- | --- |
| 8 | 프레임마다 경고 | ultralytics 8.4.184에서 `half=True` 미지원 | `app.py`, `pose_app.py`, `proximity_app.py`에서 제거 |
| 8 | 이어폰 → remote, 에어컨 → microwave | COCO 80종 밖 물체 | 임베딩 재라벨, YOLOE |
| 9 | 만세가 한팔들기로 깜빡임 | 한쪽 손목이 화면 밖 | 카메라에서 물러나 두 손이 다 보이게 |
| 9 | 검증 98.9%인데 가만히 있어도 wave | 얼굴 특징이 학습 때 항상 0 → 표준편차 ≈ 0 → 정규화 폭주 | `std.clamp(min=0.03)` (0.1은 박수도 둔해짐) + 박수 추가 녹화 |
| 9 | 행동 모델 학습 윈도 0개 | 지나가는 데 1.5~2초, 창은 80프레임 | `--window 15` |
| 10 | 렌즈 보정 RMS가 1.4~3.5 px로 흔들림 | C920 자동초점 | `v4l2-ctl`로 자동초점 끔 (USB 다시 꽂으면 초기화) |
| 10 | 위치 보정 첫 계산 172 px | 실측표와 클릭에서 기준점 번호가 바뀜 | json에서 좌표 교환 후 해당 점만 재클릭 |
| 10 | 이동 타겟이 10~14 m 밖으로 튐 | 상자를 카메라 높이에 가깝게 들어 시선이 수평 | 상자를 무릎~허리 높이로 |
| 10 | 타겟이 바닥 아래·8 m 밖 | 보정 뒤 cam0이 약 6° 돌아감 | 보정 때 화면과 비교해 손으로 다시 맞춤 |

## 안내문과 다르게 한 점

- `half=True` 제거 (위 표).
- 9단계 `train_pose.py`: 표준편차 하한 `std.clamp(min=0.03)`.
- 10단계 카메라는 `--cams 0 2`, 렌즈 보정은 태블릿 체커보드(칸 22 mm, `--square 0.022`).
- 10단계 시나리오는 공간이 좁아 4 m 대신 3 m 자리에서 진행, 이동 타겟은 카트 대신 사람이 든 갈색 상자.

## 한계와 다음 개선

- 카메라 높이 1.35~1.41 m, 두 시선 각도 23°로 권장(2.2~2.5 m, 60~90°)보다 낮고 좁아 한 카메라 추정 오차가 컸다.
- 몸을 돌리거나 앉고 일어설 때 0~1초짜리 가짜 ID가 생겨 불필요한 근접 알람이 났다. 짧게 사는 ID를 알람에서 빼는 것이 다음 개선이다.
- 행동 모델은 검증 데이터가 적어 점수로 판단하기 어렵다.

학습된 모델(.pt), 카메라 보정 결과(.npz), 녹화 영상은 저장소에 넣지 않았습니다.
