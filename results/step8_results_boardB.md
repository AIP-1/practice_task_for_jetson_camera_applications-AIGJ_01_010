# 8단계 결과 (보드 B, C920 웹캠)

## 8-1 기준선: YOLO만 (yolo11n, conf 0.5)

- 검출 속도: 15.0 fps (1280x720)
- 참고: README의 `half=True`는 ultralytics 8.4.184에서 프레임마다 경고가 떠서 제거함

| 물체 | COCO 80종에 있음? | yolo 라벨 | 결과 |
| --- | --- | --- | --- |
| 휴대폰 | 있음 | cell phone | 정답 |
| 노트북 | 있음 | laptop | 정답 |
| 의자 | 있음 | chair | 정답 |
| 마우스 | 있음 | mouse | 정답 |
| 음료컵 | 있음 | cup | 정답 |
| 모니터 | 있음 (tv) | monitor로 보고됨 (COCO에선 tv) | 확인 필요 |
| 우산 | 있음 | umbrella | 정답 |
| 무선 이어폰 | 없음 | remote | 오인식 |
| 천장 에어컨 | 없음 | microwave | 오인식 |

COCO에 있는 물체는 모두 맞혔고, COCO에 없는 무선 이어폰·천장 에어컨은 가장 비슷한 COCO 클래스(remote, microwave)로 잘못 불렸다. YOLO는 아는 80종 중 하나를 반드시 고르기 때문이다.

## 8-2 개선 1: 임베딩 재라벨 (relabel)

- 검출 속도: 15~17 fps (재라벨은 별도 스레드라 YOLO 속도는 거의 그대로)

`app_v2.py`의 `LABELS`에 팀 테스트 물체와 헷갈리는 짝(이어폰/리모컨, 에어컨/전자레인지)을 넣고 실행.

| 물체 | yolo 라벨 | relabel 라벨 (점수) |
| --- | --- | --- |
| 무선 이어폰 | mouse (8-1에선 remote) | 무선 이어폰 ✓ |
| 모자 | suitcase | 컵 ✗ (LABELS에 모자가 없어 가장 가까운 "컵"을 고름 → hat/야구 모자 추가 후 재시험) |
| 드럼 세탁기 | microwave | 드럼 세탁기 ✓ |
| 선풍기 | clock | 천장 에어컨 ✗ (목록에 선풍기 없음) |
| 공기청정기 | cup | 물병 ✗ (목록에 공기청정기 없음) |
| 시계 | clock (정답) | 휴대폰 / 건전지 ✗ (YOLO가 맞혔는데 목록에 시계가 없어 오히려 틀어짐) |

1차 결과 정리: 목록에 있는 물체(무선 이어폰, 드럼 세탁기)는 YOLO가 틀려도 바로잡았다. 목록에 없는 물체(모자, 선풍기, 공기청정기, 시계)는 목록 중 가장 가까운 이름으로 잘못 바뀌었고, 시계처럼 YOLO가 맞힌 것도 망가졌다. → 테스트 물체를 LABELS에 추가(hat, electric fan, air purifier, wall clock)하고 재시험.

YOLO는 이어폰을 각도에 따라 remote, mouse 등 그때그때 다른 COCO 이름으로 부르지만, 재라벨은 "무선 이어폰"으로 바로잡았다. 재라벨은 LABELS 목록 안에서만 고르므로 목록에 없는 물체(모자)는 고칠 수 없다.

## 8-3 개선 2: YOLOE (yoloe-11s-seg, conf 0.25)

LABELS 영문 이름(모자·선풍기·공기청정기·시계 추가 후)을 검출 클래스로 사용.

| 물체 | yoloe 라벨 | 결과 |
| --- | --- | --- |
| 무선 이어폰 | mouse | ✗ (목록에 둘 다 있는데 마우스로 헷갈림) |
| 천장 에어컨 | ceiling air conditioner / microwave | △ (각도에 따라 오락가락) |
| 선풍기 | electric fan | ✓ |
| 컵 | cup | ✓ |
| 시계 | wall clock | ✓ |
| 드럼 세탁기 | drum washing machine | ✓ |
| 캡 모자 | umbrella | ✗ |

- 검출 속도: 약 13 fps (YOLO 15 fps보다 조금 느림)

## 8-4 세 방식 비교

| 물체 | yolo | relabel | yoloe |
| --- | --- | --- | --- |
| 무선 이어폰 | remote / mouse ✗ | 무선 이어폰 ✓ | mouse ✗ |
| 천장 에어컨 | microwave ✗ | (미확인) | ceiling air conditioner / microwave △ |
| 드럼 세탁기 | microwave ✗ | 드럼 세탁기 ✓ | drum washing machine ✓ |
| 선풍기 | clock ✗ | 천장 에어컨 ✗ (목록 추가 전) | electric fan ✓ |
| 시계 | clock ✓ | 휴대폰 / 건전지 ✗ (목록 추가 전) | wall clock ✓ |
| 모자 | suitcase ✗ | 컵 ✗ (목록 추가 전) | umbrella ✗ |
| 컵 | cup ✓ | | cup ✓ |

| 모드 | 검출 fps |
| --- | --- |
| yolo | 15 |
| relabel | 15~17 |
| yoloe | 13 |

정리: relabel은 목록에 적어 둔 물체를 정확히 바로잡지만(이어폰, 드럼 세탁기), 목록 밖 물체는 엉뚱한 이름으로 바꾼다. yoloe는 선풍기·시계처럼 COCO에 없는 물체도 이름으로 바로 찾지만, 작고 비슷한 물체(이어폰↔마우스)나 모자는 아직 헷갈린다.

강의실 서비스에 쓴다면: 찾을 물체 목록이 정해져 있으므로 YOLOE로 후보를 찾고(13 fps로 실시간에 충분), 작은 물체처럼 YOLOE가 헷갈리는 경우는 임베딩 재라벨로 한 번 더 확인하는 조합이 가장 낫다. 이 조합은 README 10단계의 타겟 찾기 방식과 같다.
