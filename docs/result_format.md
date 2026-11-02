# 결과 형식 문서 (AIGJ_01_010 박수연)

다른 프로그램이 결과를 받아 갈 수 있도록, 실제로 있는 조회 주소와 응답 형식만 적었습니다.

- 세 앱 모두 FastAPI 웹 서버로, 보드의 8000번 포트에서 HTTP로 결과를 내보냅니다. 한 번에 하나만 띄웁니다.
- **MQTT 발행은 구현하지 않았습니다.** 안내문의 `aiot/...` 토픽(재실 인원, 물품 점검, 키오스크, 근접 알람)은 없습니다.
- 영상은 보드 안 MJPEG 스트림으로만 봅니다. 사람 이미지는 세 앱 모두 저장하지 않습니다(8단계는 person 클래스를 크롭하지 않음). 8단계는 사람 외 물체 크롭을 보드의 `crops/<모드>/`에 최대 1000개까지 저장하고, 넘으면 오래된 것부터 지웁니다.
- 아래 JSON의 숫자는 형식을 보여 주는 예시이고 측정값이 아닙니다.

## 8단계 물체 추출·검색 — `app.py`

| 주소 | 방식 | 내용 |
| --- | --- | --- |
| `/` | GET | 웹 화면 (실시간 영상, 텍스트 검색, 최근 추출) |
| `/stream` | GET | MJPEG 영상 (`multipart/x-mixed-replace`) |
| `/search?q=<문장>&k=12` | GET | 텍스트로 물체 검색, 점수 높은 순 k개 |
| `/recent?n=12` | GET | 최근 추출한 물체 n개와 처리 속도 |
| `/crops/<모드>/<파일>` | GET | 크롭 이미지 |

물체 항목 (`/search`, `/recent`의 `items`):

```json
{"id": 57, "cls": "mouse", "label": "무선 이어폰", "lscore": 0.71, "conf": 0.62,
 "ts": 1761550000.1, "img": "/crops/relabel/000123_mouse.jpg", "score": 0.583}
```

| 필드 | 뜻 |
| --- | --- |
| `cls`, `conf` | 검출기(YOLO/YOLOE)가 붙인 클래스와 신뢰도 |
| `label`, `lscore` | 최종 라벨과 그 점수 (relabel 모드에서는 임베딩으로 고른 라벨) |
| `ts` | 추출 시각 (유닉스 초) |
| `img` | 크롭 이미지 주소 (`/crops/...`, 같은 서버에서 GET) |
| `score` | 검색 질의와의 유사도 (`/search`에서만) |

`/recent` 응답: `{"mode": "relabel", "stats": {"fps": ..., "embed_ms": ...}, "total": ..., "items": [...]}` (`embed_ms`는 크롭 1개당 임베딩 시간)

## 9단계 동작·행동 인식 — `pose_app.py`

| 주소 | 방식 | 내용 |
| --- | --- | --- |
| `/` | GET | 웹 화면 (영상, 현재 상태, 동작별 횟수, 관심도 통계, 동작 기록) |
| `/stream` | GET | MJPEG 영상 |
| `/events` | GET | 현재 상태와 기록 (화면이 1초마다 조회) |

```json
{"mode": "smooth +behavior", "stats": {"fps": 14.8, "people": 1},
 "counts": {"만세": 3, "WAVE": 2},
 "events": [{"ts": 1761550000.1, "who": "#4", "label": "WAVE"}],
 "beh": {"지나감": 5, "힐끗 봄": 3, "관심 있음": 3, "머무름": 3},
 "interest_rate": 0.21,
 "now": {"#4": "만세 / 관심 있음"}}
```

| 필드 | 뜻 |
| --- | --- |
| `counts` | 동작별 누적 횟수 |
| `events` | 최근 동작·행동 기록 20개 (최신이 앞) |
| `beh` | 행동 패턴별 사람 수 (`--behavior`일 때) |
| `interest_rate` | 관심 있음 / 전체 사람 수 |
| `now` | 지금 화면에 있는 사람별 동작(과 행동 상태) |

## 10단계 근접 알람 — `prox/proximity_app.py`

| 주소 | 방식 | 내용 |
| --- | --- | --- |
| `/` | GET | 웹 화면 (카메라 2대 영상, 평면도, 사람 표, 알람 기록) |
| `/stream/0`, `/stream/1` | GET | 카메라별 MJPEG 영상 |
| `/state` | GET | 타겟·사람 위치와 알람 기록 (화면이 0.5초마다 조회) |
| `/retarget` | POST | 타겟 다시 찾기 |

`/state` 주요 필드:

| 필드 | 뜻 |
| --- | --- |
| `stats` | `{"fps", "skew_ms" (두 카메라 시간 차), "mode"}` |
| `target` | `{"xyz": [x, y, z] m, "how": 찾은 방법, "trail", "v": 속도, "lost"}` (`--moving`일 때 `trail`·`v`) |
| `people[]` | 사람별 `id`, 바닥 좌표 `x`·`y` (m), 타겟까지 거리 `d` (m), `trend`, `zone` (near / mid / far), 접근 속도 `closing` (m/s), `ttc` (s), 자세 `pose`, 추정 방식 `kind` (stereo/single), `danger` |
| `events[]` | 최근 25개 `{"ts", "who", "label"}`. 알람은 `label`이 `알람`으로 시작 (`알람: 근접 1.0m`, `알람: 이탈 3.1m`, `알람: 충돌 위험 TTC 1.9초 (접근 1.0m/s, 2.0m)`). 그 밖에 `등장`, `퇴장 (머문 시간 N초)`, 타겟 `찾음`·`다시 찾음` 기록 |
| `room`, `near`, `far`, `cams`, `cam_pos` | 방 범위, 알람 거리 기준, 카메라 수와 위치 |
