# Jetson 실습 - EmbeddingGemma 2 멀티모달 임베딩

Oct 8, 2026 · @안녕

## 개요

Jetson Orin Nano Super 8GB에서 EmbeddingGemma 2를 설치하고 텍스트·영상·음성 임베딩을 하나의 벡터 공간에서 검색해 보는 실습이다. 이어서 도메인 미세 조정, 웹캠 물체 검색, 사람 동작 인식, 카메라 2대 접근 감지까지 강의실에서 바로 쓸 수 있는 응용으로 확장한다.

| 항목 | 값 |
| --- | --- |
| 모델 | [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) (Apache 2.0) |
| 크기 | 텍스트 270M · +비전 440M · +오디오 570M · 전체 740M |
| 출력 차원 | 768 (Matryoshka: 512 / 256 / 128) |
| 컨텍스트 | 8,192 토큰 (모든 모달리티 공유) |
| 권장 dtype | bfloat16 또는 float32 (float16 금지) |
| 입력 조건 | 오디오 16 kHz 모노, 비디오 기본 1 fps 샘플링 |

준비물:

- Jetson Orin Nano Super 8GB, JetPack 6.x, NVMe SSD 권장 (모델 + 캐시 약 3GB)
- 인터넷 연결 (Hugging Face 모델 다운로드)
- 테스트 파일: 이미지 3\~5장, 10초 내외 동영상 1개, 음성/소리 WAV 2\~3개 (동영상은 RTSP 카메라, 음성은 5단계 명령으로 직접 생성 가능)

**단계별 흐름**

실습은 "환경 → 임베딩 기초 → 카메라 응용" 순서로 쌓여 간다. 1\~6단계는 모든 실습의 공통 기반이고, 7단계는 텍스트 검색 성능을 높이는 갈래, 8\~10단계는 웹캠을 쓰는 응용 갈래다. 8단계 이후 실습은 모두 "문제 확인 → 개선" 순서로 진행한다.

| 단계 | 하는 일 | 결과물 | 다음 단계와의 연결 |
| --- | --- | --- | --- |
| 1. 환경 준비 | SSD 마운트, 전원 모드·스왑, JetPack에 맞는 GPU용 PyTorch 설치 | GPU 연산이 확인된 venv (`~/eg2`) | 이후 모든 단계가 이 venv와 torch 위에서 돈다. torch가 깨지지 않게 버전을 고정해 둔다 |
| 2. 패키지·모델 설치 | sentence-transformers 설치, EmbeddingGemma 2 내려받기, 공통 로더 작성 | `eg2.py`의 `load()` (필요한 인코더만 골라 로드) | 3\~5단계가 `load()`로 텍스트·비전·오디오를 골라 쓴다 |
| 3. 텍스트 테스트 | 질의/문서 프롬프트로 한국어 검색, 차원(768/256/128)별 비교 | "질의 ↔ 문서" 유사도 검색의 기본형 | 같은 검색 방식을 4·5단계는 이미지·음성으로, 7단계는 미세 조정으로 확장한다 |
| 4. 영상 테스트 | 이미지·동영상 임베딩, 텍스트로 이미지 검색 | 텍스트와 이미지가 같은 벡터 공간에 있다는 확인 | 8단계가 이 능력으로 웹캠 크롭을 검색하고 라벨을 다시 붙인다 |
| 5. 음성 테스트 | 16kHz 모노 변환, 음성·소리 임베딩과 검색 | 텍스트로 소리를 찾는 예제 | 멀티모달 확인을 마무리하고, 6단계에서 3\~5단계 성능을 한데 측정한다 |
| 6. 성능 측정 | 구성별 RAM·지연·GPU 사용률 측정, 자주 겪는 오류 정리 | 구성별 자원 사용량 표 | 8\~10단계에서 YOLO·EmbeddingGemma를 함께 올릴 때 메모리·fps 판단 기준이 된다 |
| 7. 도메인 미세 조정 | 강의실 도메인 질의-문서 쌍으로 텍스트 모델 추가 학습, 학습 전후 Recall·nDCG 비교 | `models/eg2-classroom` (텍스트 전용) | 텍스트 RAG 갈래의 끝. 이미지 검색에는 쓸 수 없어 8단계부터는 원본 모델을 쓴다. 학습·평가 방법은 9-6에서 다시 쓴다 |
| 8. 웹캠 물체 추출·검색 | YOLO로 물체 검출 → 크롭 임베딩 → 웹에서 실시간 영상·텍스트 검색. COCO 밖 물체의 오인식을 확인하고 임베딩 재라벨·YOLOE로 개선 | `app.py` (카메라 스레드 + FastAPI 웹 구조) | 9·10단계가 같은 웹 구조를 쓰고, 10단계는 YOLOE + EmbeddingGemma 조합으로 타겟을 찾는다 |
| 9. 사람 동작·행동 인식 | 관절 17개로 만세·손 흔들기·악수 판정, 시간 안정화, 관심도(정면 응시) 패턴, GRU 학습 | `pose_app.py`, `pose_model.py`, `train_pose.py` | 관절 좌표와 추적·시간 창 판정 방식을 10단계가 그대로 쓴다 (어깨 중심 = 사람 위치) |
| 10. 카메라 2대 접근 감지 | 렌즈·위치 보정 후 어깨 중심 삼각측량으로 사람의 실제 위치(m)를 구하고, 타겟과의 거리·접근 속도로 알람. 이동 타겟은 칼만 필터와 TTC로 확장 | `prox/` (보정 도구 2개 + `proximity_app.py`) | 최종 응용. 알람을 MQTT로 내보내면 환경 모니터링 에이전트·팀 프로젝트와 연결된다 |

시간이 부족하면 1→2→3→4→6 뒤 목적에 맞는 갈래를 고른다. 텍스트 RAG가 목표면 7단계, 카메라 응용이 목표면 8→9→10단계로 간다. 5단계(음성)는 다른 단계의 전제가 아니어서 건너뛰어도 된다. Q&A에는 실제 설치·실행 중 나온 오류를 단계 순서대로 모아 두었다.

## 1단계 환경 준비

Jetson에서는 PyPI 기본 torch가 CPU 전용이라 GPU가 잡히지 않는다. 반드시 JetPack에 맞는 CUDA 휠을 먼저 설치한다.

**0. NVMe SSD 마운트** (microSD로 부팅하고 SSD를 데이터용으로 쓸 때). `lsblk`에서 `nvme0n1`의 MOUNTPOINT가 `/`이면 이미 SSD로 부팅 중이니 건너뛴다. 아래 `parted`·`mkfs`는 SSD의 기존 데이터를 모두 지운다.

```bash
lsblk -f                                   # nvme0n1 확인

# 파티션·포맷 (새 SSD일 때만)
sudo parted /dev/nvme0n1 --script mklabel gpt mkpart primary ext4 0% 100%
sudo mkfs.ext4 -L ssd /dev/nvme0n1p1

# 마운트 + 부팅 시 자동 마운트
sudo mkdir -p /ssd
UUID=$(sudo blkid -s UUID -o value /dev/nvme0n1p1)
echo "UUID=$UUID /ssd ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
sudo mount -a
sudo chown $USER:$USER /ssd
df -h /ssd                                 # 용량 확인

# Hugging Face 모델 캐시를 SSD로
mkdir -p /ssd/hf
echo 'export HF_HOME=/ssd/hf' >> ~/.bashrc && source ~/.bashrc
```

`nofail` 옵션은 SSD가 빠져 있어도 부팅이 멈추지 않게 한다. 재부팅 후 `df -h /ssd`로 자동 마운트를 확인한다.

1. JetPack·CUDA 버전 확인

```bash
cat /etc/nv_tegra_release        # R36.x = JetPack 6, R38.x = JetPack 7
nvcc --version                    # CUDA 12.x
```

2. 최대 성능 모드 설정

```bash
sudo nvpmodel -q                  # 현재 모드 확인
sudo nvpmodel -m 0                # MAXN SUPER (모드 번호는 -q 출력으로 확인)
sudo jetson_clocks
```

3. 스왑 8GB 추가 (풀 모델 로드·비디오 디코딩 시 OOM 방지)

```bash
# SSD를 마운트했으면 /ssd/swapfile, 아니면 /swapfile
SWAP=/ssd/swapfile
sudo fallocate -l 8G $SWAP && sudo chmod 600 $SWAP
sudo mkswap $SWAP && sudo swapon $SWAP
echo "$SWAP none swap sw 0 0" | sudo tee -a /etc/fstab
free -h
```

4. 시스템 패키지와 가상환경

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip ffmpeg libsndfile1 libopenblas-dev
python3 -m venv ~/eg2 && source ~/eg2/bin/activate
pip install -U pip wheel
```

5. Jetson용 PyTorch 설치 (jetson-ai-lab 인덱스, JetPack 6 / CUDA 12.6 기준)

JetPack 버전에 따라 설치 경로가 다르다. `python3 -V`가 3.12이면 JetPack 7(Ubuntu 24.04), 3.10이면 JetPack 6(Ubuntu 22.04)이다.

| JetPack | Ubuntu / Python | CUDA | torch 인덱스 |
| --- | --- | --- | --- |
| 7.x (R38) | 24.04 / 3.12 | 13.x | `https://download.pytorch.org/whl/cu132` (공식 SBSA 휠) |
| 6.x (R36) | 22.04 / 3.10 | 12.6 | `https://pypi.jetson-ai-lab.io/jp6/cu126` (cp310 전용) |

JetPack 7 (Python 3.12):

```bash
python -m pip install --pre torch torchvision \
  --extra-index-url https://download.pytorch.org/whl/cu132
```

JetPack 6 (Python 3.10):

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

검증은 `is_available()`만으로는 부족하다. 실제 GPU 커널이 돌아야 한다.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(), torch.cuda.is_bf16_supported()); x=torch.rand(1024,1024,device='cuda'); print((x@x).sum().item())"
# 기대값: (8, 7)  True  숫자 출력. 'no kernel image' 또는 sm_87 경고가 나오면 휠이 맞지 않는 것
```

`from versions: none`은 대부분 Python 버전과 인덱스가 어긋난 경우다. `externally-managed-environment`는 venv가 활성화되지 않은 채 시스템 pip를 쓴 경우이므로 4번부터 다시 한다.

인덱스 주소는 JetPack 버전에 따라 다르다. `True`가 안 나오면 다음 단계로 넘어가지 말고 6단계의 문제 해결을 먼저 본다. 대안으로 [jetson-containers](https://github.com/dusty-nv/jetson-containers)의 PyTorch 컨테이너를 써도 된다.

## 2단계 패키지·모델 설치

핵심은 sentence-transformers를 설치할 때 Jetson torch가 PyPI torch로 덮어쓰이지 않게 막는 것이다. 설치된 torch 버전을 제약 파일로 고정한 뒤 설치한다.

```bash
source ~/eg2/bin/activate
pip freeze | grep -E '^(torch|torchvision|torchaudio)==' > ~/eg2/torch-pin.txt

pip install -c ~/eg2/torch-pin.txt -U \
  "sentence-transformers[image,audio,video]>=6.1.0" transformers accelerate \
  soundfile librosa av pillow huggingface_hub
```

모델 미리 받기 (저장 형식에 따라 1.5\~3GB, 실습 중 네트워크 대기 방지):

```bash
hf download google/embeddinggemma-2
```

설치 확인 스크립트 `check.py`:

```python
import torch, transformers, sentence_transformers as st
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("sentence-transformers", st.__version__)   # 6.1.0 이상이어야 함
assert torch.cuda.is_available(), "Jetson torch가 덮어쓰여졌는지 확인"
```

공통 로더 `eg2.py` (다음 단계들이 공유):

```python
import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "google/embeddinggemma-2"

def load(vision=False, audio=False):
    cfg = {}
    if not vision: cfg["vision_config"] = None   # 미사용 인코더는 로드하지 않음
    if not audio:  cfg["audio_config"] = None
    return SentenceTransformer(
        MODEL_ID, device="cuda",
        model_kwargs={"torch_dtype": torch.bfloat16},  # float16 금지
        config_kwargs=cfg,
    )
```

**실행 절차**

1. 실습 폴더를 만들고 venv를 활성화한다. 이후 모든 스크립트는 이 폴더에서 저장·실행한다.
2. 위 두 코드를 각각 `check.py`, `eg2.py`로 저장한다 (`nano check.py` → 붙여넣기 → Ctrl+O, Enter, Ctrl+X).
3. `check.py`를 실행해 세 줄이 모두 출력되고 assert가 통과하는지 본다.
4. `eg2.py`가 모델을 올리는지 한 줄로 확인한다. 첫 실행은 모델 로드에 수십 초가 걸린다.

```bash
source ~/eg2/bin/activate
mkdir -p ~/eg2-lab/media && cd ~/eg2-lab

nano check.py        # check.py 코드 붙여넣기
nano eg2.py          # eg2.py 코드 붙여넣기

python check.py
python -c "from eg2 import load; m = load(); print(m.encode('test').shape)"
# 기대값: (768,)
```

## 3단계 텍스트 테스트

텍스트 전용(270M)으로 로드해 한국어 질의가 알맞은 문서를 1위로 찾는지 확인한다. 질의는 `SearchQuery`, 문서는 `Document` 프롬프트를 꼭 구분해서 쓴다.

`test_text.py`:

```python
import time
from eg2 import load

model = load()  # 텍스트 전용

docs = [
    "환기가 부족하면 강의실 CO2 농도가 1000ppm을 넘어 집중력이 떨어진다.",
    "에어컨 설정 온도는 하계 26도, 동계 20도가 권장된다.",
    "조도가 300lux 미만이면 필기 작업 시 눈의 피로가 커진다.",
    "Jetson Orin Nano는 67 TOPS의 AI 성능을 제공한다.",
]
queries = ["수업 중에 졸리고 머리가 맑지 않아요", "여름에 냉방 몇 도로 맞춰야 하나요?"]

d = model.encode(docs, prompt_name="Document", normalize_embeddings=True)

for dim in [768, 256, 128]:
    t = time.perf_counter()
    q = model.encode(queries, prompt_name="SearchQuery",
                     truncate_dim=dim, normalize_embeddings=True)
    ms = (time.perf_counter() - t) * 1000
    dd = d[:, :dim]
    dd = dd / (dd ** 2).sum(1, keepdims=True) ** 0.5   # 자른 뒤 재정규화
    sim = q @ dd.T
    print(f"\n[dim={dim}] {ms:.0f} ms")
    for qi, row in zip(queries, sim):
        best = row.argmax()
        print(f"  {qi} -> {docs[best][:25]}... ({row[best]:.3f})")
```

**실행 절차**

1. `~/eg2-lab`에서 venv가 켜져 있는지 확인한다 (프롬프트 앞에 `(eg2)`).
2. 다른 터미널에서 `tegrastats`를 켜 둔다 (6단계 측정용).
3. 위 코드를 `test_text.py`로 저장하고 실행한다.
4. 출력에서 차원별 지연(ms)과 각 질의의 1위 문서를 확인하고, 6단계 측정 표의 "텍스트 전용" 행에 기록한다.

```bash
cd ~/eg2-lab && source ~/eg2/bin/activate
nano test_text.py    # 코드 붙여넣기
python test_text.py
```

출력 형식 (수치는 장비마다 다름):

```text
[dim=768] xxx ms
  수업 중에 졸리고 머리가 맑지 않아요 -> 환기가 부족하면 강의실 CO2 농도가 100... (0.xxx)
  여름에 냉방 몇 도로 맞춰야 하나요? -> 에어컨 설정 온도는 하계 26도, 동계 2... (0.xxx)
[dim=256] xx ms
  ...
```

확인 포인트:

- 첫 질의는 CO2 문장, 둘째 질의는 에어컨 문장이 모든 차원에서 1위여야 한다.
- 첫 실행은 CUDA 초기화 때문에 느리다. 지연 시간은 두 번째 차원부터 비교한다.
- 변형 과제: `prompt_name`을 빼고 돌려 점수 차이를 비교해 본다.

## 4단계 영상 테스트

비전 인코더를 켜서(440M) 텍스트 질의로 이미지와 동영상을 검색한다. 이미지는 기본 280 토큰, 비디오는 초당 1프레임으로 처리된다.

테스트 자료 준비 (`media/` 폴더):

- 이미지 3\~5장: 빈 강의실, 수업 중 강의실, 화이트보드, 모니터 등 서로 다른 장면
- 동영상 1개: 10초 내외. 없으면 IP 카메라 실습의 RTSP 스트림에서 잘라 쓴다.

```bash
mkdir -p media
# RTSP에서 10초 녹화 (주소는 실습 카메라에 맞게)
ffmpeg -rtsp_transport tcp -i rtsp://<camera>/stream -t 10 -an -c:v libx264 media/clip.mp4
# 같은 영상에서 2초마다 프레임 추출 (프레임 단위 검색용)
ffmpeg -i media/clip.mp4 -vf fps=0.5 media/frame_%02d.jpg
```

`test_vision.py`:

```python
import glob, time
from eg2 import load

model = load(vision=True)

images = sorted(glob.glob("media/*.jpg"))
t = time.perf_counter()
img_emb = model.encode([{"image": p} for p in images], normalize_embeddings=True)
print(f"이미지 {len(images)}장: {(time.perf_counter()-t)*1000:.0f} ms")

t = time.perf_counter()
vid_emb = model.encode({"video": "media/clip.mp4"}, normalize_embeddings=True)
print(f"동영상 1개: {(time.perf_counter()-t)*1000:.0f} ms")

queries = ["사람이 없는 빈 강의실", "학생들이 앉아 수업을 듣는 모습", "화이트보드에 쓴 글씨"]
q = model.encode(queries, prompt_name="SearchQuery", normalize_embeddings=True)

sim = q @ img_emb.T
for qi, row in zip(queries, sim):
    order = row.argsort()[::-1][:3]
    print(qi, "->", [(images[i].split('/')[-1], round(float(row[i]), 3)) for i in order])

print("질의-동영상 유사도:", (q @ vid_emb).round(3))
```

**실행 절차**

1. `~/eg2-lab/media`에 이미지 3\~5장을 복사하거나, 위 ffmpeg 명령으로 `clip.mp4`와 `frame_*.jpg`를 만든다.
2. 파일이 제대로 있는지 확인한다.
3. 위 코드를 `test_vision.py`로 저장하고 실행한다. 비전 인코더 로드로 텍스트 단계보다 RAM을 더 쓴다.
4. 각 질의의 상위 3개 이미지를 열어 보고 결과가 맞는지 판단한다. 지연과 RAM을 측정 표의 "+비전" 두 행에 기록한다.

```bash
cd ~/eg2-lab && source ~/eg2/bin/activate
ls -lh media/*.jpg media/*.mp4
ffprobe -v error -show_entries format=duration -of csv=p=0 media/clip.mp4   # 길이(초)

nano test_vision.py  # 코드 붙여넣기
python test_vision.py
```

출력 형식:

```text
이미지 5장: xxx ms
동영상 1개: xxxx ms
사람이 없는 빈 강의실 -> [('frame_03.jpg', 0.xxx), ...]
질의-동영상 유사도: [0.xxx 0.xxx 0.xxx]
```

동영상 단계에서 오류가 나면 `vid_emb`와 마지막 `print` 두 줄을 주석 처리하고 이미지 검색만 먼저 완료한다.

확인 포인트:

- 각 질의의 1위 이미지가 사람 눈으로 봐도 맞는지 확인한다.
- 멀티모달 검색은 768 또는 512차원을 쓴다. 128차원에서는 영상·음성 품질이 약 75%로 떨어진다.
- `{"video": ...}` 디코딩이 실패하면 위에서 추출한 프레임 이미지를 사용해 프레임 단위 검색으로 대체한다 (6단계 참고).
- 확장 과제: 텍스트와 이미지를 섞은 입력 `{"text": "3월 강의실 <|image|>", "image": ["media/frame_01.jpg"]}`을 넣어 본다.

## 5단계 음성 테스트

오디오 인코더를 켜서(570M) 텍스트 질의로 음성·소리 파일을 검색한다. 입력은 반드시 16 kHz 모노로 맞춘다.

샘플 만들기 (마이크가 없어도 가능):

```bash
sudo apt install -y espeak-ng alsa-utils
# 한국어 합성 음성
espeak-ng -v ko "강의실이 너무 덥습니다. 에어컨을 켜 주세요." -w media/raw_speech.wav
# 경보음(1 kHz 비프)과 빗소리 같은 잡음
ffmpeg -f lavfi -i "sine=frequency=1000:duration=3" media/raw_beep.wav
ffmpeg -f lavfi -i "anoisesrc=color=pink:duration=5" media/raw_noise.wav
# USB 마이크가 있으면 직접 녹음 (장치는 arecord -l로 확인)
arecord -D plughw:1,0 -f S16_LE -r 16000 -c 1 -d 5 media/raw_mic.wav

# 전부 16 kHz 모노로 변환
for f in media/raw_*.wav; do
  ffmpeg -y -i "$f" -ar 16000 -ac 1 "media/${f#media/raw_}"
done
```

`test_audio.py`:

```python
import glob, time
import soundfile as sf
from eg2 import load

model = load(audio=True)

files = sorted(f for f in glob.glob("media/*.wav") if "/raw_" not in f)
for f in files:
    info = sf.info(f)
    assert info.samplerate == 16000 and info.channels == 1, f"{f}: 16kHz 모노가 아님"

t = time.perf_counter()
aud_emb = model.encode([{"audio": f} for f in files], normalize_embeddings=True)
print(f"오디오 {len(files)}개: {(time.perf_counter()-t)*1000:.0f} ms")

queries = ["덥다고 냉방을 요청하는 목소리", "날카로운 경보음", "웅웅거리는 배경 소음"]
q = model.encode(queries, prompt_name="SearchQuery", normalize_embeddings=True)

sim = q @ aud_emb.T
for qi, row in zip(queries, sim):
    best = row.argmax()
    print(f"{qi} -> {files[best].split('/')[-1]} ({row[best]:.3f})")
```

**실행 절차**

1. 위 샘플 명령을 `~/eg2-lab`에서 실행해 `media/speech.wav`, `beep.wav`, `noise.wav`(마이크가 있으면 `mic.wav`)를 만든다.
2. 변환된 파일이 16 kHz 모노인지 확인하고, `aplay`로 들어 본다 (스피커가 있을 때).
3. 위 코드를 `test_audio.py`로 저장하고 실행한다. assert가 실패하면 해당 파일을 ffmpeg로 다시 변환한다.
4. 각 질의가 알맞은 파일을 1위로 찾는지 확인하고, 지연과 RAM을 측정 표의 "+오디오" 행에 기록한다.

```bash
cd ~/eg2-lab && source ~/eg2/bin/activate
for f in media/speech.wav media/beep.wav media/noise.wav; do
  ffprobe -v error -show_entries stream=sample_rate,channels -of csv=p=0 "$f"   # 16000,1
done
aplay media/speech.wav

nano test_audio.py   # 코드 붙여넣기
python test_audio.py
```

출력 형식:

```text
오디오 3개: xxx ms
덥다고 냉방을 요청하는 목소리 -> speech.wav (0.xxx)
날카로운 경보음 -> beep.wav (0.xxx)
웅웅거리는 배경 소음 -> noise.wav (0.xxx)
```

확인 포인트:

- 세 질의가 각각 speech, beep, noise 파일을 1위로 찾으면 성공이다.
- espeak-ng 합성음은 사람 목소리보다 점수가 낮게 나올 수 있다. 가능하면 마이크 녹음으로 비교해 본다.
- 확장 과제: `load(vision=True, audio=True)`로 전체 모델을 올려 이미지·영상·음성을 한 인덱스에 넣고 하나의 질의로 검색해 본다.

## 6단계 성능 측정과 문제 해결

각 테스트를 돌리는 동안 다른 터미널에서 `tegrastats`를 켜 두고, 아래 표를 직접 채운다. 지연은 두 번째 실행 값을 적는다.

```bash
sudo tegrastats --interval 1000 | grep -oE 'RAM [0-9]+/[0-9]+MB|GR3D_FREQ [0-9]+%'
```

| 구성 | 로드 크기 | 로드 후 RAM (MB) | 입력 | 지연 (ms) | GPU 사용률 (%) |
| --- | --- | --- | --- | --- | --- |
| 텍스트 전용 | 270M |  | 질의 2개 |  |  |
| +비전 | 440M |  | 이미지 5장 |  |  |
| +비전 | 440M |  | 동영상 10초 |  |  |
| +오디오 | 570M |  | WAV 3개 |  |  |
| 전체 | 740M |  | 혼합 |  |  |

자주 겪는 문제:

| 증상 | 원인 | 해결 |
| --- | --- | --- |
| `torch.cuda.is_available()`가 False | sentence-transformers 설치 중 PyPI torch로 덮어씀 | 1단계 5번 torch 재설치 후 2단계처럼 `-c torch-pin.txt`로 설치 |
| 임베딩에 NaN, 유사도가 이상함 | float16 사용 | `torch.bfloat16` 또는 `float32`로 변경 |
| `CUDA out of memory` 또는 프로세스 강제 종료 | 8GB 공유 메모리 부족 | 안 쓰는 인코더 끄기, 배치 나눠 encode, 스왑 확인, 다른 LLM 서버 중지 |
| `{"video": ...}`에서 디코더 오류 | aarch64용 비디오 디코딩 백엔드 미설치 | `pip install av` 후 재시도, 안 되면 ffmpeg로 추출한 프레임을 이미지로 입력 |
| `prompt_name`을 찾을 수 없다는 오류 | sentence-transformers 구버전 | `pip install -c ~/eg2/torch-pin.txt -U "sentence-transformers>=6.1.0"` |
| 오디오 검색 결과가 무작위에 가까움 | 샘플레이트·채널 불일치 | ffmpeg로 `-ar 16000 -ac 1` 변환 후 재실행 |
| 모델 다운로드가 느리거나 실패 | 네트워크·디스크 | NVMe에 `HF_HOME` 지정, `hf download`로 미리 받기 |

GPU 메모리가 빠듯하면 텍스트 전용은 llama.cpp의 [GGUF](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF) (Q8\_0, 310MB)로 대체할 수 있다.

## 7단계 도메인 미세 조정

강의실 환경 도메인의 질의-문서 쌍으로 텍스트 모델(270M)을 추가 학습하고, 학습 전후 Recall@k를 비교한다. 이미지·음성까지 포함한 미세 조정은 8GB에서 메모리가 빠듯해 PC GPU나 Colab에서 학습하고 Jetson에서는 추론만 하는 것을 권장한다. 아래 코드는 EmbeddingGemma 1의 [공식 미세 조정 가이드](https://ai.google.dev/gemma/docs/embeddinggemma/fine-tuning-embeddinggemma-with-sentence-transformers) 방식을 적용한 것이다.

**데이터 준비**

한 줄에 질의 하나와 정답 문서 하나를 넣는다. 문서 출처는 강의실 운영 FAQ, 센서 기준치 설명, 센서 로그 요약 문장 등이다. 질의는 문서마다 2\~3개씩 실제 교육생이 물을 법한 말투로 쓴다. 로컬 LLM으로 질의를 생성해도 된다.

- 목표 규모: 학습 300\~2,000쌍, 평가 50\~100쌍 (평가용 문서는 학습에 쓰지 않음)
- 같은 배치의 다른 문서가 오답으로 쓰이므로, 내용이 거의 같은 문서는 하나로 합친다.

`data/train.jsonl` (평가용 `data/eval.jsonl`도 같은 형식):

```json
{"anchor": "수업 중에 졸리고 머리가 맑지 않아요", "positive": "환기가 부족하면 강의실 CO2 농도가 1000ppm을 넘어 집중력이 떨어진다."}
{"anchor": "여름 냉방 온도 권장값", "positive": "에어컨 설정 온도는 하계 26도, 동계 20도가 권장된다."}
```

제공되는 `eval.jsonl`은 60쌍이다. 문서 20개에 문서마다 질의가 3개씩 붙어 있다. 다루는 항목은 온도·습도·PM2.5·조도·소음·TVOC·재실 인원·냉난방·환기·센서 관리·경보·에이전트 권한이다. 문서의 수치는 실습용 가상 운영 기준이다. 학습 데이터(`train.jsonl`)는 282쌍이다. 문서 94개에 문서마다 질의가 3개씩 붙어 있고, 평가 세트와는 문서·질의가 하나도 겹치지 않는다. 다루는 범위는 CO2·환기, 냉난방, 공기질, 센서·네트워크, 경보, 에이전트, 엣지 운영, 과정 운영이다.

**평가 스크립트** `eval_ir.py` (학습 전후 모두 사용):

```python
import json, sys, torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

def build_evaluator(path="data/eval.jsonl", name="classroom"):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    corpus, queries, relevant, doc_id = {}, {}, {}, {}
    for i, r in enumerate(rows):
        d = doc_id.setdefault(r["positive"], f"d{len(doc_id)}")
        corpus[d] = r["positive"]
        queries[f"q{i}"] = r["anchor"]
        relevant.setdefault(f"q{i}", set()).add(d)
    return InformationRetrievalEvaluator(
        queries, corpus, relevant, name=name,
        query_prompt_name="SearchQuery", corpus_prompt_name="Document",
        accuracy_at_k=[1, 3, 5], truncate_dim=None)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "google/embeddinggemma-2"
    model = SentenceTransformer(path, device="cuda",
        model_kwargs={"torch_dtype": torch.bfloat16},
        config_kwargs={"vision_config": None, "audio_config": None})
    res = build_evaluator()(model)
    for k in ["accuracy@1", "accuracy@3", "ndcg@10"]:
        print(k, round(res[f"classroom_cosine_{k}"], 3))
```

**학습 스크립트** `train_ft.py`:

```python
import torch
from datasets import load_dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss, MatryoshkaLoss
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from eval_ir import build_evaluator

model = SentenceTransformer("google/embeddinggemma-2", device="cuda",
    model_kwargs={"torch_dtype": torch.bfloat16},
    config_kwargs={"vision_config": None, "audio_config": None})

# gradient checkpointing은 Trainer 인자 대신 내부 HF 모델에 직접 켬다
# (Trainer 경유 시 every_n_layers TypeError 발생 → Q9)
hf = getattr(model[0], "auto_model", None) or getattr(model[0], "model", None)
hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

train = load_dataset("json", data_files="data/train.jsonl", split="train")

# MatryoshkaLoss로 감싸야 학습 후에도 256/128차원 절단이 유지됨
loss = MatryoshkaLoss(model, MultipleNegativesRankingLoss(model),
                      matryoshka_dims=[768, 512, 256, 128])

args = SentenceTransformerTrainingArguments(
    output_dir="runs/eg2-classroom",
    num_train_epochs=3,
    per_device_train_batch_size=8,        # OOM이면 4로
    gradient_accumulation_steps=2,
    learning_rate=2e-5, warmup_steps=0.1,  # Transformers v5: 비율을 float로
    bf16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES,   # 배치 내 중복 문서 방지
    prompts={"anchor": model.prompts["SearchQuery"],
             "positive": model.prompts["Document"]},
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
    logging_steps=10, report_to="none",
)

trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train,
                                     loss=loss, evaluator=build_evaluator())
trainer.train()
model.save_pretrained("models/eg2-classroom")
```

**실행 절차**

1. `~/eg2-lab/data`에 `train.jsonl`, `eval.jsonl`을 만들고 줄 수를 확인한다.
2. 다른 LLM 서버 등 GPU를 쓰는 프로세스를 멈추고, `tegrastats`를 켜 둔다.
3. 기준 모델을 먼저 평가해 값을 기록한다.
4. 학습을 실행한다. 로그의 loss가 줄어드는지, epoch마다 평가 점수가 오르는지 본다.
5. 미세 조정 모델을 평가하고 아래 표에 두 값을 비교한다.
6. 2단계 `eg2.py`의 `MODEL_ID`를 `"models/eg2-classroom"`으로 바꾸면 이후 테스트가 모두 미세 조정 모델로 돌아간다.

```bash
cd ~/eg2-lab && source ~/eg2/bin/activate
pip install -c ~/eg2/torch-pin.txt datasets
wc -l data/train.jsonl data/eval.jsonl

python eval_ir.py                          # 기준 모델
nohup python train_ft.py > train.log 2>&1 &   # 학습 (SSH 끊겨도 계속)
tail -f train.log
python eval_ir.py models/eg2-classroom     # 미세 조정 모델
```

| 모델 | accuracy@1 | accuracy@3 | ndcg@10 | 학습 시간 |
| --- | --- | --- | --- | --- |
| 기준 (google/embeddinggemma-2) |  |  |  | - |
| 미세 조정 (eg2-classroom) |  |  |  |  |

확인 포인트:

- 평가 점수가 학습 후 오히려 떨어지면 과적합이다. epoch를 1\~2로 줄이거나 learning\_rate를 1e-5로 낮춘다.
- 도메인 밖 일반 질의(3단계 테스트)도 다시 돌려 일반 성능이 크게 떨어지지 않았는지 확인한다.
- OOM이 나면 배치를 4로 줄이고 `gradient_accumulation_steps`를 4로 늘린다. 단, 배치가 작을수록 오답 수가 줄어 학습 효과도 작아진다.

## 8단계 웹캠 실시간 물체 추출과 웹 검색

USB 웹캠 영상에서 물체를 검출·추출하고, 웹 브라우저에서 실시간 영상과 텍스트 검색(예: "빨간 텐블러")을 확인한다. 실습은 **문제 확인 → 개선 1 → 개선 2** 순서로 진행한다. 먼저 YOLO만으로 실행해 COCO 80종 밖의 물체가 오인식되는 것을 확인하고, EmbeddingGemma 2 재라벨과 YOLOE 오픈 어휘 검출을 차례로 적용해 같은 물체로 결과를 비교한다. 코드는 `app.py` 하나이고 `--mode`로 방식을 바꾼다.

처리 흐름 (스레드 2개 + 웹 서버):

1. **캡처·검출 스레드**: 웹캠 프레임 → YOLO 추적(track) → 박스를 그린 JPEG를 웹 스트림용으로 갱신. 새 추적 ID의 물체(또는 30초마다)만 잘라 큐에 넣는다.
2. **임베딩 스레드**: 큐의 크롭을 최대 8개씩 묶어 512차원 벡터로 변환하고, 크롭 이미지와 함께 메모리 인덱스에 저장 (최대 1,000개, 오래된 것부터 삭제).
3. **웹 서버(FastAPI)**: `/` 화면, `/stream` MJPEG 실시간 영상, `/search?q=` 텍스트 검색, `/recent` 최근 추출 목록.

개인정보 보호를 위해 `person` 클래스는 기본으로 크롭·저장하지 않는다. 사람은 영상의 박스로만 표시된다.

**설치**

```bash
cd ~/eg2-lab && source ~/eg2/bin/activate
pip install -c ~/eg2/torch-pin.txt ultralytics fastapi "uvicorn[standard]"
sudo apt install -y v4l-utils
v4l2-ctl --list-devices             # 웹캠이 /dev/video0 인지 확인
v4l2-ctl -d /dev/video0 --list-formats-ext | head -20   # MJPG 1280x720 지원 확인
```

ultralytics는 AGPL-3.0 라이선스다. 교육·내부 실습에는 문제없지만, 서비스로 외부 공개할 때는 라이선스를 확인한다.

**코드** `app.py`:

```python
import os, time, threading, collections, argparse
import cv2, numpy as np, torch, uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer

# ---------------- 실행 모드 ----------------
# yolo    : YOLO 클래스 그대로 사용 (문제 확인용 기준선)
# relabel : YOLO 박스 + EmbeddingGemma 2 재라벨 (개선 1)
# yoloe   : 텍스트로 클래스를 지정하는 YOLOE 검출기 (개선 2)
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["yolo", "relabel", "yoloe"], default="relabel")
ap.add_argument("--conf", type=float, default=None, help="검출 신뢰도 (기본: 모드별 값)")
ap.add_argument("--port", type=int, default=8000)
args = ap.parse_args()
MODE = args.mode

# ---------------- 설정 ----------------
CAM, W, H     = 0, 1280, 720          # /dev/video0
CONF = args.conf if args.conf is not None else {"yolo": 0.5, "relabel": 0.35, "yoloe": 0.25}[MODE]
MIN_SIDE      = 48                    # 크롭 최소 크기(px)
REFRESH_S     = 30                    # 같은 추적 ID 재임베딩 간격(초)
SKIP_CLASSES  = {"person"}            # 크롭·저장하지 않을 클래스
DIM, MAX_ITEMS = 512, 1000
MODEL_ID      = "google/embeddinggemma-2"   # 7단계 모델은 텍스트 전용이라 여기서는 원본 사용
CROP_DIR      = f"crops/{MODE}"       # 모드별로 크롭을 따로 저장해 나중에 비교

# YOLO(COCO 80종)에 없는 물체는 비슷한 클래스로 오인식된다 (세탁기→oven, 전지 포장→cell phone).
# relabel: 크롭 임베딩과 아래 라벨 문구의 유사도로 라벨을 다시 붙인다.
# yoloe  : 아래 영문 라벨을 검출 클래스 자체로 사용한다.  (영문 = 영상 표시, 한글 = 웹 표시)
LABELS = [
    ("drum washing machine", "드럼 세탁기"), ("top-load washing machine", "통돌이 세탁기"),
    ("oven", "오븐"), ("microwave", "전자레인지"), ("refrigerator", "냉장고"),
    ("cell phone", "휴대폰"), ("battery pack", "원통형 전지 포장"), ("AA battery", "건전지"),
    ("laptop", "노트북"), ("monitor", "모니터"), ("keyboard", "키보드"), ("mouse", "마우스"),
    ("cup", "컵"), ("water bottle", "물병"), ("book", "책"), ("backpack", "가방"), ("chair", "의자"),
]
LABEL_MIN     = 0.20                  # 이 점수 미만이면 YOLO 라벨 유지 (웹의 점수를 보고 조정)
os.makedirs(CROP_DIR, exist_ok=True)

if MODE == "yoloe":
    from ultralytics import YOLOE
    names = [en for en, _ in LABELS]
    det = YOLOE("yoloe-11s-seg.pt")     # 첫 실행 시 자동 다운로드
    det.set_classes(names, det.get_text_pe(names))
else:
    det = YOLO("yolo11n.pt")          # 첫 실행 시 자동 다운로드
KO = dict(LABELS)                     # 영문 -> 한글
print(f"[mode={MODE}] conf={CONF}")
emb = SentenceTransformer(MODEL_ID, device="cuda",
        model_kwargs={"torch_dtype": torch.bfloat16},
        config_kwargs={"audio_config": None})   # 텍스트+비전(440M)

with torch.inference_mode():
    label_vecs = np.asarray(emb.encode([f"a photo of a {en}" for en, _ in LABELS],
        prompt_name="SearchQuery", truncate_dim=DIM, normalize_embeddings=True), dtype=np.float32)

frame_lock, emb_lock, idx_lock = threading.Lock(), threading.Lock(), threading.Lock()
latest_jpeg = None
crop_q = collections.deque(maxlen=64)
items = []          # {id, cls, conf, ts, img, vec}
seen = {}           # track_id -> 마지막 임베딩 시각
track_label = {}    # track_id -> (영문 라벨, 한글 라벨, 점수)
stats = {"fps": 0.0, "embed_ms": 0.0}

def capture_loop():
    global latest_jpeg
    cap = cv2.VideoCapture(CAM, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    assert cap.isOpened(), "웹캠을 열 수 없음 (Q11 참고)"
    t0, n = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1); continue
        r = det.track(frame, persist=True, conf=CONF, half=True, verbose=False)[0]
        now = time.time()
        if r.boxes.id is not None:
            for (x1, y1, x2, y2), tid, c, cf in zip(r.boxes.xyxy.int().tolist(),
                    r.boxes.id.int().tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                name = det.names[c]
                if name in SKIP_CLASSES or min(x2 - x1, y2 - y1) < MIN_SIDE:
                    continue
                if now - seen.get(tid, 0) < REFRESH_S:
                    continue
                seen[tid] = now
                crop_q.append((tid, name, cf, now, frame[y1:y2, x1:x2].copy()))
        vis = frame.copy()
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
        for (x1, y1, x2, y2), tid, c, cf in zip(r.boxes.xyxy.int().tolist(), ids,
                r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
            yolo = det.names[c]
            en = track_label[tid][0] if (MODE == "relabel" and tid in track_label) else yolo
            color = (0, 200, 255) if en != yolo else (0, 255, 0)   # 재라벨된 박스는 주황색
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{en} {cf:.2f}", (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        n += 1
        if now - t0 >= 1:
            stats["fps"], t0, n = n / (now - t0), now, 0
        cv2.putText(vis, f"{stats['fps']:.1f} fps | indexed {len(items)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with frame_lock:
                latest_jpeg = buf.tobytes()

def embed_loop():
    seq = 0
    while True:
        if not crop_q:
            time.sleep(0.05); continue
        batch = [crop_q.popleft() for _ in range(min(8, len(crop_q)))]
        paths = []
        for tid, name, cf, ts, crop in batch:
            seq += 1
            p = f"{CROP_DIR}/{seq:06d}_{name}.jpg"
            cv2.imwrite(p, crop)
            paths.append(p)
        t = time.perf_counter()
        with emb_lock:
            vecs = emb.encode([{"image": p} for p in paths],
                              truncate_dim=DIM, normalize_embeddings=True)
        stats["embed_ms"] = (time.perf_counter() - t) * 1000 / len(paths)
        vecs = np.asarray(vecs, dtype=np.float32)
        sims = vecs @ label_vecs.T
        with idx_lock:
            for (tid, name, cf, ts, _), p, v, sim in zip(batch, paths, vecs, sims):
                j = int(sim.argmax())
                if MODE != "relabel":            # yolo / yoloe: 검출기 라벨 그대로
                    en, ko, ls = name, KO.get(name, name), float(sim[j])
                elif sim[j] >= LABEL_MIN:
                    en, ko, ls = LABELS[j][0], LABELS[j][1], float(sim[j])
                else:
                    en, ko, ls = name, name, float(sim[j])
                track_label[tid] = (en, ko, ls)
                items.append({"id": tid, "cls": name, "label": ko, "lscore": round(ls, 3),
                              "conf": round(cf, 2), "ts": ts, "img": "/" + p, "vec": v})
            while len(items) > MAX_ITEMS:
                old = items.pop(0)
                try: os.remove(old["img"][1:])
                except OSError: pass

app = FastAPI()
app.mount("/crops", StaticFiles(directory="crops"), name="crops")   # /crops/<mode>/<파일>

def public(it, score=None):
    d = {k: it[k] for k in ("id", "cls", "label", "lscore", "conf", "ts", "img")}
    if score is not None: d["score"] = round(float(score), 3)
    return d

@app.get("/stream")
def stream():
    def gen():
        while True:
            with frame_lock:
                f = latest_jpeg
            if f:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
            time.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/search")
def search(q: str, k: int = 12):
    with idx_lock:
        snap = list(items)
    if not snap:
        return []
    with emb_lock:
        qv = emb.encode(q, prompt_name="SearchQuery", truncate_dim=DIM, normalize_embeddings=True)
    s = np.stack([it["vec"] for it in snap]) @ np.asarray(qv, dtype=np.float32)
    return [public(snap[i], s[i]) for i in s.argsort()[::-1][:k]]

@app.get("/recent")
def recent(n: int = 12):
    with idx_lock:
        snap = items[-n:][::-1]
    return {"mode": MODE, "stats": stats, "total": len(items), "items": [public(it) for it in snap]}

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>Jetson 물체 검색</title>
<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}
img.live{width:100%;max-width:960px;border-radius:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px}
.card{background:#222;padding:6px;border-radius:6px;font-size:12px}
.card img{width:100%;height:110px;object-fit:contain;background:#000}
input{font-size:16px;padding:6px;width:60%}button{font-size:16px;padding:6px 12px}</style>
<h2>실시간 영상 <span id=mode></span></h2><img class=live src="/stream">
<h2>텍스트로 물체 검색</h2>
<form onsubmit="go();return false"><input id=q placeholder="예: 빨간 컬, 열려 있는 노트북"><button>검색</button></form>
<div id=res class=grid></div>
<h2>최근 추출 <span id=st></span></h2><div id=rec class=grid></div>
<script>
const card=it=>`<div class=card><img src="${it.img}"><div><b>${it.label}</b> (${it.lscore})<br>YOLO: ${it.cls} ${it.conf}`+
 (it.score!==undefined?`<br>검색 점수 <b>${it.score}</b>`:'')+`<br>${new Date(it.ts*1000).toLocaleTimeString()}</div></div>`;
async function go(){const q=document.getElementById('q').value;if(!q)return;
 const r=await (await fetch('/search?q='+encodeURIComponent(q))).json();
 document.getElementById('res').innerHTML=r.map(card).join('')||'결과 없음';}
async function poll(){const r=await (await fetch('/recent')).json();
 document.getElementById('mode').textContent=`[mode: ${r.mode}]`;
 document.getElementById('st').textContent=`(총 ${r.total}개, 검출 ${r.stats.fps.toFixed(1)} fps, 임베딩 ${r.stats.embed_ms.toFixed(0)} ms/개)`;
 document.getElementById('rec').innerHTML=r.items.map(card).join('');}
setInterval(poll,3000);poll();
</script>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE

if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=embed_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
```

| `--mode` | 검출기 | 라벨 결정 | 기본 conf | 목적 |
| --- | --- | --- | --- | --- |
| `yolo` | yolo11n (COCO 80종) | YOLO 클래스 그대로 | 0.5 | 문제 확인 (기준선) |
| `relabel` | yolo11n | 크롭 임베딩 ↔ `LABELS` 문구 유사도 | 0.35 | 개선 1 |
| `yoloe` | YOLOE (`LABELS` 영문을 클래스로 지정) | 검출기 클래스 | 0.25 | 개선 2 |

세 모드 모두 크롭을 EmbeddingGemma 2로 임베딩하므로 텍스트 검색은 항상 동작한다. 크롭은 `crops/<mode>/`에 따로 쌓여 나중에 비교할 수 있다.

**테스트 물체 준비** (세 모드에서 같은 물체를 같은 위치·조명으로 보여준다)

| 구분 | 물체 예시 | 기대 |
| --- | --- | --- |
| COCO에 있는 물체 | 컵, 노트북, 휴대폰, 책 | 세 모드 모두 정답 |
| COCO에 없는 물체 | 드럼 세탁기(실물 또는 모니터에 띄운 사진), 원통형 전지 포장, 건전지 | yolo에서 오인식, 개선 후 정답 |

### 8-1 기준선: YOLO만으로 실행 (문제 확인)

1. Jetson IP를 확인하고 `yolo` 모드로 실행한다. 첫 실행에서 `yolo11n.pt`를 받는다.
2. 브라우저에서 `http://<Jetson IP>:8000`에 접속해 제목 옆에 `[mode: yolo]`가 보이는지 확인한다.
3. 테스트 물체를 하나씩 5초 이상 보여주고, 영상 박스의 라벨을 8-4 표의 yolo 열에 적는다. 박스가 아예 안 잡히면 "미검출"로 적는다.
4. 검색창에 "세탁기", "건전지"를 넣고, 결과 카드의 YOLO 라벨과 이미지를 비교한다.
5. Ctrl+C로 종료한다.

```bash
cd ~/eg2-lab && source ~/eg2/bin/activate
hostname -I                  # Jetson IP
sudo ufw status              # active이면: sudo ufw allow 8000/tcp
python app.py --mode yolo
# [mode=yolo] conf=0.5 → Uvicorn running on http://0.0.0.0:8000
```

확인할 문제:

- 드럼 세탁기가 `oven`, 원통형 전지 포장이 `cell phone`처럼 엉뚱한 클래스로 표시된다. YOLO는 학습한 80종 중 하나를 반드시 고르기 때문이다.
- 그런데 "세탁기" 검색은 맞는 크롭을 찾을 수 있다. 검색은 라벨이 아니라 이미지 임베딩을 비교하기 때문이다. 라벨과 검색이 서로 다른 경로라는 점을 확인한다.
- 건전지처럼 작거나 낯선 물체는 박스조차 안 잡힐 수 있다. 박스가 없으면 크롭도, 검색도 되지 않는다.

### 8-2 개선 1: 임베딩 재라벨 (relabel)

YOLO는 박스 위치만 쓰고, 라벨은 크롭 임베딩과 `LABELS` 문구("a photo of a ...")의 유사도로 다시 정한다. 추가 학습이 필요 없고, 박스가 잡히는 물체라면 라벨을 바로잡을 수 있다. 검출 신뢰도도 0.35로 낮춰 애매한 물체의 박스를 더 확보한다.

1. `relabel` 모드로 다시 실행하고 같은 물체를 같은 순서로 보여준다.
2. 재라벨된 박스는 영상에 주황색으로 표시된다. 웹 카드의 "라벨(점수)"과 원래 YOLO 클래스를 8-4 표에 적는다.
3. 점수를 보고 `LABEL_MIN`(기본 0.20)을 조정한다. 맞는 라벨의 점수보다 조금 낮게 둔다.
4. 인식하고 싶은 물체가 더 있으면 `LABELS`에 (영문, 한글) 쌍으로 추가하고 다시 실행한다. 헷갈리는 짝(오븐/세탁기, 휴대폰/전지 포장)을 함께 넣어야 올바르게 갈린다.

```bash
python app.py --mode relabel
```

확인할 점: 8-1에서 틀렸던 라벨이 바로잡혔는지, COCO 물체(컵, 노트북)는 여전히 맞는지 본다. 한계도 확인한다. YOLO가 박스를 못 잡은 물체는 여기서도 여전히 미검출이다.

### 8-3 개선 2: YOLOE 오픈 어휘 검출 (yoloe)

검출기 자체를 텍스트로 클래스를 지정하는 YOLOE로 바꾼다. `LABELS`의 영문 이름이 그대로 검출 클래스가 되므로, 8-2에서 박스가 안 잡혔던 물체도 찾을 수 있다. 첫 실행에서 YOLOE 가중치와 텍스트 인코더를 내려받는다. 모델 파일명과 API는 ultralytics 버전에 따라 다를 수 있어, 오류가 나면 설치된 버전의 YOLOE 문서를 확인한다.

1. `yoloe` 모드로 실행하고 같은 물체를 보여준다.
2. 8-1·8-2에서 "미검출"이던 물체에 박스가 생기는지 본다.
3. 화면 상단 fps가 8-2보다 얼마나 떨어지는지 기록한다. 오검출이 많으면 `--conf 0.35`로 올린다.

```bash
python app.py --mode yoloe
python app.py --mode yoloe --conf 0.35   # 오검출이 많을 때
```

### 8-4 결과 비교

| 물체 | yolo 라벨 | relabel 라벨 (점수) | yoloe 라벨 | 비고 |
| --- | --- | --- | --- | --- |
| 컵 |  |  |  |  |
| 노트북 |  |  |  |  |
| 휴대폰 |  |  |  |  |
| 드럼 세탁기 |  |  |  |  |
| 원통형 전지 포장 |  |  |  |  |
| 건전지 |  |  |  |  |

| 모드 | 검출 fps | 임베딩 ms/개 | RAM (MB, tegrastats) |
| --- | --- | --- | --- |
| yolo |  |  |  |
| relabel |  |  |  |
| yoloe |  |  |  |

정리할 내용: 세 방식의 정확도와 속도 차이, 그리고 강의실 서비스에는 어떤 방식을 쓸지 근거와 함께 한두 문장으로 적는다.

확인 포인트:

- 검출 fps가 10 미만이면 해상도를 `W, H = 640, 480`으로 낮추거나, 크롭이 너무 많이 쌓일 때 `REFRESH_S`를 60으로 늘린다.
- 같은 추적 ID는 30초에 한 번만 저장된다. 카메라 밖으로 나갔다 들어오면 새 ID로 다시 저장된다.
- 확장 과제: yoloe 박스에 relabel을 함께 적용하거나, 검색 점수가 0.3 미만인 결과를 숨기는 필터를 넣어 본다.

## 9단계 사람 동작 인식 (만세·한팔들기·손 흔들기·악수)

웹캠 영상에서 사람의 관절 17개를 찾고(YOLO11n-pose), 관절 좌표의 위치 관계와 시간 패턴으로 동작을 판정해 웹에서 확인한다. 8단계와 같이 **문제 확인(9-1) → 개선(9-2) → 학습 데이터 수집(9-3) → 행동 패턴(9-5) → 학습 모델(9-6)** 순서로 진행한다. 사람 이미지는 저장하지 않고, 저장하는 것은 동작 기록과(선택 시) 정규화된 관절 좌표뿐이다.

| 동작 | 판정 규칙 | 종류 |
| --- | --- | --- |
| 만세 | 양쪽 손목이 어깨보다 어깨 너비의 0.3배 이상 위 | 정지 자세 |
| 한팔들기 | 한쪽 손목만 위 기준을 만족 | 정지 자세 |
| 손흔들기 | 한팔들기 상태에서 1.5초 동안 손목이 좌우로 2번 이상 방향 전환 | 시간 패턴 (9-2부터) |
| 악수 | 두 사람의 손목이 어깨 너비의 0.5배 이내, 높이는 가슴\~허리 | 2인 자세 |

이미지 좌표는 y가 작을수록 화면 위쪽이다. 거리가 달라도 같은 기준이 되도록 모든 거리는 어깨 너비로 나눠 비교한다.

**설치**: 8단계에서 설치한 패키지로 충분하다. `yolo11n-pose.pt`는 첫 실행 때 자동으로 내려받는다. EmbeddingGemma는 쓰지 않아 메모리가 적게 든다.

**코드** `pose_app.py`:

```python
import time, threading, collections, argparse, csv
import cv2, numpy as np, uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO
from pose_model import (KP_CONF, NOSE, LS, RS, LW, RW, body, facing_score,
                        frame_features, load_model, predict)

# ---------------- 실행 옵션 ----------------
# --raw          : 프레임마다 판정한 결과를 그대로 표시 (9-1 문제 확인용)
# (기본)         : 0.6초 다수결 안정화 + 손 흔들기(시간 패턴) 판정 (9-2 개선)
# --record F.csv : 관절 특징을 CSV로 저장 (9-3 학습 데이터 수집)
# --behavior     : 정면 응시·이동 속도로 행동 패턴(지나감/힐끗 봄/관심 있음/머무름) 판정 (9-5)
# --model M.pt   : 학습한 GRU 모델로 판정 (9-6). 여러 번 지정 가능 (동작용, 행동용)
ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, default=0)
ap.add_argument("--conf", type=float, default=0.5)
ap.add_argument("--raw", action="store_true")
ap.add_argument("--record", default=None, help="예: data/wave.csv")
ap.add_argument("--behavior", action="store_true")
ap.add_argument("--model", action="append", default=[], help="예: models/action.pt")
ap.add_argument("--port", type=int, default=8000)
args = ap.parse_args()
MODE = "raw" if args.raw else "smooth"

W, H = 1280, 720
SMOOTH_S = 0.6        # 다수결 창(초)
WAVE_S = 1.5          # 손 흔들기 판정 창(초)
BEH_WIN = 15.0        # 행동 패턴 판정 창(초)
FACE_TH = 0.6         # 정면으로 볼 facing 점수
LOOK_S, INTEREST_S, LINGER_S = 1.0, 3.0, 10.0
MOVE_TH = 0.5         # 이동 판정 속도 (어깨 너비/초)
ML_TH = 0.7           # 모델 확률 기준
EN = {"만세": "BOTH ARMS UP", "한팔들기": "ONE ARM UP", "손흔들기": "WAVE", "악수": "HANDSHAKE",
      "기본": "-", "지나감": "PASSING", "힐끗 봄": "GLANCE", "관심 있음": "INTERESTED",
      "머무름": "LINGER", "관찰 중": "..."}
BEH_RANK = {"관찰 중": 0, "지나감": 1, "머무름": 2, "힐끗 봄": 3, "관심 있음": 4}

det = YOLO("yolo11n-pose.pt")         # 첫 실행 시 자동 다운로드
models = [load_model(p) for p in args.model]
SEQ_LEN = max([m["window"] for m in models], default=1)
print(f"[pose mode={MODE}] conf={args.conf} behavior={args.behavior} "
      f"models={[(m['task'], m['labels']) for m in models]}")

frame_lock, ev_lock = threading.Lock(), threading.Lock()
latest_jpeg = None
tracks = {}
pairs = {}            # (tid_a, tid_b) -> deque[(t, bool)]
pair_on = set()
events = collections.deque(maxlen=50)
counts = collections.Counter()
beh_counts = collections.Counter()   # 사람(추적 ID) 단위 최종 행동
stats = {"fps": 0.0, "people": 0}
rec = None
if args.record:
    rec_f = open(args.record, "a", newline="")
    rec = csv.writer(rec_f)

def new_track(now):
    return {"raw": collections.deque(maxlen=90), "wave": collections.deque(maxlen=90),
            "beh": collections.deque(maxlen=600), "feat": collections.deque(maxlen=SEQ_LEN),
            "label": "기본", "bstate": "관찰 중", "best": "관찰 중", "ml": {},
            "first": now, "seen": now, "prev": None, "look": 0.0, "n": 0}

def pose_label(k, b):
    (cx, cy), sw = b
    raised = lambda w: k[w, 2] > KP_CONF and k[w, 1] < cy - 0.3 * sw
    lu, ru = raised(LW), raised(RW)
    if lu and ru:
        return "만세"
    if lu or ru:
        return "한팔들기"
    return "기본"

def is_wave(hist):
    """올린 손목의 x 좌표(어깨 너비로 정규화)가 좌우로 2번 이상 방향을 바꾸면 손 흔들기"""
    xs = [x for _, x in hist]
    if len(xs) < 8:
        return False
    changes, last_dir, anchor = 0, 0, xs[0]
    for x in xs[1:]:
        d = x - anchor
        if abs(d) > 0.15:
            cur = 1 if d > 0 else -1
            if last_dir and cur != last_dir:
                changes += 1
            last_dir, anchor = cur, x
    return changes >= 2

def behavior(st, now):
    """정면 연속 시간·누적 시간, 이동 여부, 체류 시간으로 행동 패턴 판정"""
    h = [x for x in st["beh"] if now - x[0] <= BEH_WIN]
    if len(h) < 5:
        return "관찰 중"
    run = 0.0                                   # 끝에서부터 정면 연속 시간
    for i in range(len(h) - 1, 0, -1):
        if h[i][1] < FACE_TH:
            break
        run = h[-1][0] - h[i - 1][0]
    look = sum(h[i][0] - h[i - 1][0] for i in range(1, len(h)) if h[i][1] >= FACE_TH)
    moving = np.median([s for t, _, s in h if now - t < 2.0]) > MOVE_TH
    if run >= INTEREST_S and not moving:
        return "관심 있음"
    if moving and LOOK_S <= look < INTEREST_S:
        return "힐끗 봄"
    if moving:
        return "지나감"
    if now - st["first"] >= LINGER_S:
        return "머무름"
    return "관찰 중"

def majority(dq, now, window):
    vals = [v for t, v in dq if now - t <= window]
    if not vals:
        return None, 0.0
    v, n = collections.Counter(vals).most_common(1)[0]
    return v, n / len(vals)

def log_event(now, who, label):
    with ev_lock:
        events.appendleft({"ts": now, "who": who, "label": label})
        counts[label] += 1

def close_track(tid, st, now):
    """추적이 끝난 사람의 요약을 남긴다"""
    if not args.behavior or st["best"] == "관찰 중":
        return
    dwell = st["seen"] - st["first"]
    with ev_lock:
        events.appendleft({"ts": now, "who": f"#{tid}",
                           "label": f"요약: 체류 {dwell:.0f}초, 정면 {st['look']:.0f}초 → {st['best']}"})
        beh_counts[st["best"]] += 1

def handshake_pairs(kps, ids, now):
    """두 사람의 손목이 가까이(어깨 너비 0.5배 이내) 있고 높이가 가슴~허리 사이면 악수 후보"""
    found = set()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = kps[i], kps[j]
            ba, bb = body(a), body(b)
            if ba is None or bb is None:
                continue
            sw = (ba[1] + bb[1]) / 2
            best = 1e9
            for wa in (LW, RW):
                for wb in (LW, RW):
                    if a[wa, 2] > KP_CONF and b[wb, 2] > KP_CONF:
                        mid_y = (a[wa, 1] + b[wb, 1]) / 2
                        if ba[0][1] < mid_y < ba[0][1] + 2.5 * ba[1]:
                            best = min(best, np.linalg.norm(a[wa, :2] - b[wb, :2]))
            key = tuple(sorted((ids[i], ids[j])))
            pairs.setdefault(key, collections.deque(maxlen=60)).append((now, best < 0.5 * sw))
            found.add(key)
    return found

def update_person(tid, k, now):
    st = tracks.setdefault(tid, new_track(now))
    dt = now - st["seen"]
    st["seen"] = now
    b = body(k)
    if b is None:
        return st
    raw = pose_label(k, b)
    fa = facing_score(k)
    spd = 0.0 if st["prev"] is None else float(np.linalg.norm(b[0] - st["prev"]) / b[1] / max(dt, 1e-3))
    st["prev"] = b[0]
    st["raw"].append((now, raw))
    st["beh"].append((now, fa, spd))
    if fa >= FACE_TH and st["n"] > 0:
        st["look"] += dt
    st["n"] += 1
    if raw in ("한팔들기", "만세"):           # 손 흔들기용: 올라간 손목의 x (어깨 중심 기준, 어깨 너비 단위)
        w = LW if k[LW, 1] < k[RW, 1] else RW
        st["wave"].append((now, (k[w, 0] - b[0][0]) / b[1]))
    feat = frame_features(k, b, fa, spd)
    st["feat"].append(feat)
    if rec:
        rec.writerow([round(now, 3), tid, raw, round(fa, 3), round(spd, 3)] + feat[:51].round(3).tolist())

    # ---- 동작 (규칙) ----
    if MODE == "raw":
        label = raw
    else:
        v, frac = majority(st["raw"], now, SMOOTH_S)
        label = v if frac >= 0.6 else st["label"]
        recent = [(t, x) for t, x in st["wave"] if now - t <= WAVE_S]
        if label == "한팔들기" and is_wave(recent):
            label = "손흔들기"

    # ---- 학습 모델 (5프레임마다) ----
    if models and st["n"] % 5 == 0:
        seq = list(st["feat"])
        for m in models:
            if len(seq) >= m["window"]:
                lab, p = predict(m, seq[-m["window"]:])
                st["ml"][m["task"]] = (lab, p) if p >= ML_TH else None
    ml_act = st["ml"].get("action")
    if ml_act:
        label = "기본" if ml_act[0] == "idle" else ml_act[0]

    if label != st["label"]:
        if label != "기본":
            log_event(now, f"#{tid}", label)
        st["label"] = label

    # ---- 행동 패턴 ----
    if args.behavior:
        bs = behavior(st, now)
        ml_beh = st["ml"].get("behavior")
        if ml_beh:
            bs = ml_beh[0]
        if bs != st["bstate"]:
            if bs != "관찰 중":
                log_event(now, f"#{tid}", bs)
            st["bstate"] = bs
        if BEH_RANK.get(bs, 1) > BEH_RANK.get(st["best"], 0):
            st["best"] = bs
    return st

def capture_loop():
    global latest_jpeg
    cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    assert cap.isOpened(), "웹캠을 열 수 없음 (Q11 참고)"
    t0, n = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1); continue
        r = det.track(frame, persist=True, conf=args.conf, half=True, verbose=False)[0]
        now = time.time()
        vis = r.plot(labels=False)        # 박스 + 관절 뼈대
        ids, kps, boxes = [], [], []
        if r.boxes.id is not None and r.keypoints is not None:
            ids = r.boxes.id.int().tolist()
            kps = np.concatenate([r.keypoints.xy.cpu().numpy(),
                                  r.keypoints.conf.cpu().numpy()[..., None]], axis=-1)
            boxes = r.boxes.xyxy.int().tolist()
        stats["people"] = len(ids)

        for tid, k, (x1, y1, x2, y2) in zip(ids, kps, boxes):
            st = update_person(tid, k, now)
            txt = f"#{tid} {EN.get(st['label'], st['label'])}"
            if args.behavior:
                txt += f" | {EN.get(st['bstate'], st['bstate'])}"
            color = (0, 200, 255) if st["label"] != "기본" or st["bstate"] == "관심 있음" else (0, 255, 0)
            cv2.putText(vis, txt, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # ---- 2인 동작 (악수) ----
        active = handshake_pairs(kps, ids, now) if len(ids) >= 2 else set()
        for key in list(pairs):
            if key not in active:
                pairs.pop(key); pair_on.discard(key); continue
            if MODE == "raw":
                on = pairs[key][-1][1]
            else:
                v, frac = majority(pairs[key], now, SMOOTH_S)
                on = bool(v) and frac >= 0.6
            if on and key not in pair_on:
                log_event(now, f"#{key[0]}+#{key[1]}", "악수")
            (pair_on.add if on else pair_on.discard)(key)
        for a, b_ in pair_on:
            if a in ids and b_ in ids:
                ba, bb = boxes[ids.index(a)], boxes[ids.index(b_)]
                x, y = (ba[0] + bb[2]) // 2 - 80, min(ba[1], bb[1]) + 30
                cv2.putText(vis, EN["악수"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)

        # 3초 이상 안 보인 사람은 추적 종료 + 요약
        for tid in [t for t, s in tracks.items() if now - s["seen"] > 3]:
            close_track(tid, tracks.pop(tid), now)

        n += 1
        if now - t0 >= 1:
            stats["fps"], t0, n = n / (now - t0), now, 0
            if rec:
                rec_f.flush()
        cv2.putText(vis, f"{stats['fps']:.1f} fps | {MODE} | people {stats['people']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with frame_lock:
                latest_jpeg = buf.tobytes()

app = FastAPI()

@app.get("/stream")
def stream():
    def gen():
        while True:
            with frame_lock:
                f = latest_jpeg
            if f:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
            time.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/events")
def get_events():
    with ev_lock:
        total = sum(beh_counts.values())
        return {"mode": MODE + (" +behavior" if args.behavior else "") + (" +model" if models else ""),
                "stats": stats, "counts": dict(counts), "events": list(events)[:20],
                "beh": dict(beh_counts),
                "interest_rate": round(beh_counts["관심 있음"] / total, 2) if total else None,
                "now": {f"#{t}": s["label"] + (f" / {s['bstate']}" if args.behavior else "")
                        for t, s in tracks.items()}}

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>Jetson 동작 인식</title>
<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}
img{width:100%;max-width:960px;border-radius:8px}table{border-collapse:collapse;margin-top:8px}
td,th{border:1px solid #444;padding:4px 10px;font-size:14px}.chip{display:inline-block;background:#333;
border-radius:12px;padding:4px 10px;margin:2px}</style>
<h2>실시간 영상 <span id=mode></span></h2><img src="/stream">
<h2>현재 상태</h2><div id=now></div>
<h2>동작별 횟수</h2><div id=cnt></div>
<h2>관심도 통계 (사람 단위)</h2><div id=beh></div>
<h2>동작 기록</h2><table><thead><tr><th>시각</th><th>대상</th><th>동작</th></tr></thead><tbody id=ev></tbody></table>
<script>
const chips=o=>Object.entries(o).map(([k,v])=>`<span class=chip>${k} ${v}</span>`).join('')||'-';
async function poll(){const r=await (await fetch('/events')).json();
 document.getElementById('mode').textContent=`[${r.mode}] ${r.stats.fps.toFixed(1)} fps, ${r.stats.people}명`;
 document.getElementById('now').innerHTML=Object.entries(r.now).map(([k,v])=>`<span class=chip>${k}: ${v}</span>`).join('')||'사람 없음';
 document.getElementById('cnt').innerHTML=chips(r.counts);
 document.getElementById('beh').innerHTML=chips(r.beh)+(r.interest_rate!==null?` <b>관심 비율 ${(r.interest_rate*100).toFixed(0)}%</b>`:'');
 document.getElementById('ev').innerHTML=r.events.map(e=>`<tr><td>${new Date(e.ts*1000).toLocaleTimeString()}</td><td>${e.who}</td><td>${e.label}</td></tr>`).join('');}
setInterval(poll,1000);poll();
</script>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE

if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
```

**공통 모듈** `pose_model.py` (`pose_app.py`와 같은 폴더에 둔다. 관절 특징 계산과 9-6의 GRU 모델):

```python
"""9단계 공통: 관절 특징 계산과 GRU 동작 분류 모델 (pose_app.py, train_pose.py에서 함께 사용)"""
import numpy as np
import torch
import torch.nn as nn

KP_CONF = 0.5
# COCO 17 관절: 0 코, 1/2 눈, 3/4 귀, 5/6 어깨, 7/8 팔꿈치, 9/10 손목, 11/12 엉덩이
NOSE, EYE_L, EYE_R, EAR_L, EAR_R, LS, RS, LW, RW, LH, RH = 0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12
N_FEAT = 34 + 17 + 2      # 정규화 좌표 34 + 관절 신뢰도 17 + 정면 점수 + 이동 속도

def body(k):
    """어깨 중심과 어깨 너비. 어깨가 안 보이면 None"""
    if k[LS, 2] < KP_CONF or k[RS, 2] < KP_CONF:
        return None
    c = (k[LS, :2] + k[RS, :2]) / 2
    sw = max(abs(k[LS, 0] - k[RS, 0]), 20.0)   # 측면에서 0에 가까워지는 것 방지
    return c, sw

def facing_score(k):
    """1에 가까울수록 카메라 정면. 코·두 눈이 보이고 코가 두 귀(없으면 두 눈) 가운데 있으면 정면"""
    if k[NOSE, 2] < KP_CONF or k[EYE_L, 2] < KP_CONF or k[EYE_R, 2] < KP_CONF:
        return 0.0
    if k[EAR_L, 2] > KP_CONF and k[EAR_R, 2] > KP_CONF:
        l, r = k[EAR_L, 0], k[EAR_R, 0]
    else:
        l, r = k[EYE_L, 0], k[EYE_R, 0]
    yaw = abs(k[NOSE, 0] - (l + r) / 2) / max(abs(l - r), 1.0)   # 0 = 정면
    return float(max(0.0, 1 - 2 * yaw))

def frame_features(k, b, facing, speed):
    """한 프레임의 특징 벡터 (N_FEAT,)"""
    c, sw = b
    norm = ((k[:, :2] - c) / sw).flatten()
    return np.concatenate([norm, k[:, 2], [facing, speed]]).astype(np.float32)

class PoseGRU(nn.Module):
    def __init__(self, n_feat, n_cls, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_cls)

    def forward(self, x):            # x: (B, T, F)
        _, h = self.gru(x)
        return self.fc(h[-1])

def load_model(path, device="cuda"):
    ck = torch.load(path, map_location=device)
    m = PoseGRU(N_FEAT, len(ck["labels"]), ck["hidden"]).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    ck["model"] = m
    ck["mean"] = torch.tensor(ck["mean"], device=device)
    ck["std"] = torch.tensor(ck["std"], device=device)
    return ck

@torch.inference_mode()
def predict(ck, seq, device="cuda"):
    """seq: (T, F) numpy → (라벨, 확률)"""
    x = torch.tensor(np.asarray(seq), device=device)[None]
    x = (x - ck["mean"]) / ck["std"]
    p = torch.softmax(ck["model"](x), -1)[0]
    i = int(p.argmax())
    return ck["labels"][i], float(p[i])
```

### 9-1 프레임 단위 판정 (문제 확인)

1. `--raw`로 실행하고 `http://<Jetson IP>:8000`에 접속한다. 제목 옆에 `[raw]`가 보여야 한다.
2. 카메라 정면 2\~3m에서 상반신이 보이게 선다. 기본 자세 → 만세 → 한팔들기 → 손 흔들기를 각 5초씩 한다.
3. 두 사람이 카메라에 측면이 보이게 마주 서서 악수를 3초 유지한다.
4. "동작별 횟수"와 실제 횟수를 9-4 표에 적는다.

```bash
cd ~/eg2-lab && source ~/eg2/bin/activate
nano pose_app.py            # 코드 붙여넣기
python pose_app.py --raw
```

확인할 문제:

- 팔을 올리거나 내리는 순간과 관절이 흔들리는 프레임마다 라벨이 깜빡이고, 동작 횟수가 실제보다 크게 늘어난다.
- 손 흔들기는 "한팔들기"로만 나온다. 한 프레임만 보면 흔드는 손과 든 손을 구분할 수 없기 때문이다.
- 악수는 손이 맞닿기 전후에도 켜졌다 꺼졌다를 반복한다.

### 9-2 시간 정보 활용 (개선)

옵션 없이 실행하면 두 가지가 추가된다. 첫째, 최근 0.6초 판정 중 60% 이상인 라벨만 확정해 깜빡임을 없앤다. 둘째, 올린 손목의 좌우 움직임을 1.5초 동안 지켜보고 손 흔들기를 따로 판정한다.

1. 옵션 없이 다시 실행하고 9-1과 같은 순서로 동작한다.
2. 라벨이 안정적으로 유지되는지, 손 흔들기가 `WAVE`로 따로 잡히는지 본다.
3. 횟수를 9-4 표에 적고, 동작 후 라벨이 바뀌기까지의 지연(약 0.3\~0.6초)도 느껴 본다.
4. 필요하면 `SMOOTH_S`(안정성 ↔ 반응 속도)와 `is_wave`의 0.15(흔드는 폭) 값을 바꿔 비교한다.

```bash
python pose_app.py
```

남는 한계: 카메라를 옆으로 보면 어깨 너비가 줄어 기준이 흔들리고, 팔이 몸에 가려지면 관절 신뢰도가 떨어져 판정이 빠진다. 박수·팔짱처럼 규칙으로 적기 어려운 동작도 늘어난다. 이때는 규칙 대신 학습 모델이 필요하다.

### 9-3 학습 데이터 수집 (확장)

`--record`를 켜면 프레임마다 한 줄씩 `시각, 추적 ID, 규칙 라벨, 정면 점수, 이동 속도, 정규화 좌표 34개, 관절 신뢰도 17개`가 CSV에 쌓인다. 좌표는 어깨 중심을 원점으로, 어깨 너비를 1로 정규화돼 있다. 이미지는 저장하지 않는다. 이 데이터로 9-6에서 모델을 학습한다.

1. **파일 이름이 곧 정답 라벨이다.** 동작은 `data/action/`, 행동은 `data/behavior/`에 라벨별 파일로 녹화한다. 규칙 라벨 열은 비교용으로만 쓴다.
2. 한 번에 한 가지 동작만 30초 이상, 여러 사람이 각도·거리를 바꿔 가며 반복한다. 녹화 중 화면에는 그 동작을 하는 사람만 있어야 한다.
3. 아무 동작도 안 하는 `idle.csv`를 꼭 함께 녹화한다. 이것이 없으면 모델이 모든 자세를 어떤 동작으로든 분류한다.

```bash
mkdir -p data/action data/behavior
# 동작: 라벨당 30초 이상
python pose_app.py --record data/action/idle.csv
python pose_app.py --record data/action/wave.csv
python pose_app.py --record data/action/clap.csv          # 규칙에 없는 새 동작
python pose_app.py --record data/action/arms_crossed.csv
# 행동: 라벨당 10회 이상 카메라 앞을 지나가거나 멈춰 서서 바라본다
python pose_app.py --record data/behavior/passing.csv
python pose_app.py --record data/behavior/glance.csv
python pose_app.py --record data/behavior/interested.csv
wc -l data/action/*.csv data/behavior/*.csv
```

### 9-4 결과 비교

| 동작 (실제 횟수) | 9-1 raw 카운트 | 9-2 smooth 카운트 | 비고 (오판·미검출) |
| --- | --- | --- | --- |
| 만세 (3회) |  |  |  |
| 한팔들기 (3회) |  |  |  |
| 손흔들기 (3회) |  |  |  |
| 악수 (3회) |  |  |  |
| 측면에서 만세 (3회) |  |  |  |

### 9-5 행동 패턴 인식 (관심도)

한 순간의 자세가 아니라, 한 사람(추적 ID)의 몇 초\~수십 초 기록을 보고 판정한다. 판정은 3개 층으로 나뉜다.

| 층 | 하는 일 | 코드 |
| --- | --- | --- |
| 1. 프레임 특징 | 정면 점수(코가 두 귀/눈 가운데 있는 정도), 이동 속도(어깨 너비/초) | `facing_score()`, `update_person()` |
| 2. 추적 이력 | 추적 ID별로 최근 15초의 (시각, 정면 점수, 속도)를 쌓음 | `st["beh"]` |
| 3. 패턴 규칙 | 이력으로 상태를 정하고, 바뀔 때 이벤트와 사람별 요약을 남김 | `behavior()`, `close_track()` |

| 패턴 | 조건 (기본값) |
| --- | --- |
| 지나감 | 이동 중 (속도 > 0.5), 정면을 본 시간 1초 미만 |
| 힐끗 봄 | 이동 중, 정면을 본 시간 1\~3초 |
| 관심 있음 | 정면을 3초 이상 연속으로 보고, 거의 멈춰 있음 |
| 머무름 | 10초 이상 멈춰 있지만 정면은 보지 않음 |

사람이 화면에서 3초 이상 사라지면 "요약: 체류 12초, 정면 4초 → 관심 있음" 같은 한 줄이 남고, 웹의 "관심도 통계"에 사람 단위로 집계된다. 한 사람의 최종 패턴은 머문 동안 도달한 가장 높은 단계(관심 있음 > 힐끗 봄 > 머무름 > 지나감)다.

1. 카메라를 게시물이나 화면 바로 위, 눈높이 가까이에 둔다. 머리 방향을 재는 것이므로 카메라와 볼 대상이 가까울수록 "봤다"에 가까워진다.
2. `--behavior`로 실행한다.
3. 네 가지 시나리오를 각각 3번씩 연기한다: 그냥 지나가기, 걸으면서 2초 보기, 멈춰 서서 5초 보기, 옆을 보며 15초 서 있기.
4. 웹의 요약 줄과 관심도 통계를 보고 아래 표에 적는다. 자주 틀리면 `FACE_TH`, `INTEREST_S`, `MOVE_TH`를 조정한다.

```bash
python pose_app.py --behavior
```

| 시나리오 (3회씩) | 기대 패턴 | 맞은 횟수 | 틀린 경우 나온 패턴 |
| --- | --- | --- | --- |
| 그냥 지나감 | 지나감 |  |  |
| 걸으며 2초 봄 | 힐끗 봄 |  |  |
| 멈춰서 5초 봄 | 관심 있음 |  |  |
| 옆을 보며 15초 서 있음 | 머무름 |  |  |

한계: 시선이 아니라 머리 방향이다. 3\~4m를 넘으면 눈·귀 관절의 신뢰도가 떨어져 정면 판정이 불안정해진다. 정해 둔 기준값은 사람마다 걸음 속도·자세가 달라 현장에서 자주 어긋난다. 이 기준을 데이터에서 배우는 것이 9-6이다.

### 9-6 학습으로 성능 개선 (GRU 시퀀스 분류)

규칙은 사람이 미리 정한 임계값이고, 학습은 그 경계를 데이터에서 찾는다. 9-3에서 모은 프레임 특징 53개를 일정 길이(윈도)로 끊어, 작은 GRU가 윈도 전체를 보고 라벨을 고르도록 학습한다. 규칙으로 적기 어려운 박수·팔짱 같은 동작도 녹화만 하면 추가할 수 있다.

| 작업 | 윈도 | 라벨 예시 | 필요한 데이터 |
| --- | --- | --- | --- |
| 동작 (`--task action`) | 30프레임 (약 1.5초) | idle, wave, clap, arms\_crossed | 라벨당 30초 이상, 3명 이상 |
| 행동 (`--task behavior`) | 80프레임 (약 4초) | passing, glance, interested | 라벨당 10회 이상 |

**학습 스크립트** `train_pose.py`:

```python
"""9-6: --record로 모은 CSV로 GRU 동작/행동 분류 모델 학습
파일 이름(확장자 제외)이 정답 라벨이다. 예) data/action/wave.csv → "wave"
사용법:
  python train_pose.py --data data/action --task action   --window 30 --out models/action.pt
  python train_pose.py --data data/behavior --task behavior --window 80 --stride 10 --out models/behavior.pt
"""
import argparse, csv, glob, os, collections
import numpy as np, torch, torch.nn as nn
from pose_model import PoseGRU, N_FEAT

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--task", choices=["action", "behavior"], required=True)
ap.add_argument("--window", type=int, default=30, help="시퀀스 길이(프레임)")
ap.add_argument("--stride", type=int, default=5)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--val", type=float, default=0.2, help="각 시퀀스 뒷부분을 검증용으로 사용")
ap.add_argument("--out", required=True)
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

def load_sequences(path):
    """추적 ID별로 시간순 정렬, 0.5초 이상 끊기면 다른 시퀀스로 나눈다"""
    by_tid = collections.defaultdict(list)
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) != 5 + 51:
                continue
            ts, tid = float(row[0]), row[1]
            fa, spd = float(row[3]), float(row[4])
            feat = np.array([float(x) for x in row[5:]] + [fa, spd], dtype=np.float32)
            by_tid[tid].append((ts, feat))
    seqs = []
    for rows in by_tid.values():
        rows.sort(key=lambda r: r[0])
        cur = [rows[0][1]]
        for (t0, _), (t1, f) in zip(rows, rows[1:]):
            if t1 - t0 > 0.5:
                seqs.append(cur); cur = []
            cur.append(f)
        seqs.append(cur)
    return [np.stack(s) for s in seqs if len(s) >= args.window]

def windows(seq):
    return [seq[i:i + args.window] for i in range(0, len(seq) - args.window + 1, args.stride)]

files = sorted(glob.glob(os.path.join(args.data, "*.csv")))
labels = [os.path.splitext(os.path.basename(p))[0] for p in files]
assert len(labels) >= 2, "라벨(CSV 파일)이 2개 이상 필요"
Xtr, ytr, Xva, yva = [], [], [], []
for ci, p in enumerate(files):
    for seq in load_sequences(p):
        cut = int(len(seq) * (1 - args.val))      # 시간순 분할: 앞 80% 학습, 뒤 20% 검증 (겹침 방지)
        tr, va = windows(seq[:cut]), windows(seq[cut:])
        Xtr += tr; ytr += [ci] * len(tr); Xva += va; yva += [ci] * len(va)
    print(f"{labels[ci]:>15}: 학습 {ytr.count(ci)}, 검증 {yva.count(ci)} 윈도")
assert Xtr and Xva, "윈도가 부족함: 녹화 시간을 늘리거나 --window를 줄일 것"

Xtr, Xva = torch.tensor(np.stack(Xtr)), torch.tensor(np.stack(Xva))
ytr, yva = torch.tensor(ytr), torch.tensor(yva)
mean = Xtr.reshape(-1, N_FEAT).mean(0)
std = Xtr.reshape(-1, N_FEAT).std(0) + 1e-6
norm = lambda x: ((x - mean) / std).to(dev)

model = PoseGRU(N_FEAT, len(labels), args.hidden).to(dev)
w = 1.0 / torch.bincount(ytr, minlength=len(labels)).float().clamp(min=1)
loss_fn = nn.CrossEntropyLoss(weight=(w / w.sum() * len(labels)).to(dev))   # 클래스 불균형 보정
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

best, best_state = -1, None
for ep in range(args.epochs):
    model.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(perm), 64):
        idx = perm[i:i + 64]
        x = norm(Xtr[idx]) + 0.05 * torch.randn(len(idx), args.window, N_FEAT, device=dev)  # 약한 잡음 증강
        loss = loss_fn(model(x), ytr[idx].to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(norm(Xva)).argmax(-1).cpu()
    acc = (pred == yva).float().mean().item()
    if acc > best:
        best, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if ep % 5 == 0 or ep == args.epochs - 1:
        print(f"epoch {ep:3d}  loss {loss.item():.3f}  val acc {acc:.3f}")

model.load_state_dict(best_state)
with torch.no_grad():
    pred = model(norm(Xva)).argmax(-1).cpu()
cm = np.zeros((len(labels), len(labels)), dtype=int)
for t, p in zip(yva.tolist(), pred.tolist()):
    cm[t, p] += 1
print(f"\n최고 검증 정확도 {best:.3f}\n혼동 행렬 (행=정답, 열=예측)")
print(" " * 16 + "".join(f"{l[:8]:>9}" for l in labels))
for l, row in zip(labels, cm):
    print(f"{l:>15} " + "".join(f"{v:>9}" for v in row))

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
torch.save({"state": best_state, "labels": labels, "task": args.task, "window": args.window,
            "hidden": args.hidden, "mean": mean.tolist(), "std": std.tolist()}, args.out)
print(f"\n저장: {args.out}  (task={args.task}, window={args.window}, labels={labels})")
```

**실행 절차**

1. 9-3 방법으로 `data/action/`과 `data/behavior/`를 모은다 (라벨당 최소 2개 파일, `idle` 포함).
2. 동작 모델을 학습하고, 출력되는 검증 정확도와 혼동 행렬을 확인한다. Jetson GPU에서 수 분이면 끝난다.
3. 행동 모델을 학습한다. 한 번 지나가는 데 4초가 안 걸리면 `--window`를 줄인다.
4. 두 모델을 넣고 실행해 9-2·9-5(규칙)와 같은 시나리오로 비교한다. 모델 확률이 0.7(`ML_TH`) 미만이면 규칙 결과가 그대로 쓰인다.

```bash
python train_pose.py --data data/action --task action --window 30 --out models/action.pt
python train_pose.py --data data/behavior --task behavior --window 80 --stride 10 --out models/behavior.pt

python pose_app.py --behavior --model models/action.pt --model models/behavior.pt
```

| 비교 항목 | 규칙 (9-2/9-5) | 학습 모델 (9-6) |
| --- | --- | --- |
| 손흔들기 인식 (9-4 시나리오) |  |  |
| 측면에서 만세 |  |  |
| 박수·팔짱 (규칙 없음) | 불가 |  |
| 힐끗 봄 / 관심 있음 구분 (9-5 표) |  |  |
| 처음 본 사람에 대한 정확도 |  |  |

학습 효과를 제대로 보려면:

- **낯선 사람으로 평가한다.** 스크립트의 검증은 같은 녹화의 뒷부분이라 점수가 높게 나온다. 학습에 참여하지 않은 사람이 실시간으로 해 봐야 실제 성능이다.
- **혼동 행렬에서 자주 헷갈리는 쌍**(예: wave↔idle)을 찾아 그 라벨만 더 녹화한다. 데이터를 늘리는 것이 모델을 키우는 것보다 효과가 크다.
- **녹화와 실행 조건을 맞춘다.** 윈도가 프레임 수라 fps가 달라지면 같은 동작의 길이가 달라진다. 해상도·카메라 위치를 녹화 때와 같게 둔다.

확인 포인트:

- 검출 fps가 낮으면 `W, H = 640, 480`으로 낮춘다. 손 흔들기 판정에는 최소 10 fps 이상이 필요하다 (1.5초 창에 8프레임 이상).
- 사람이 가려졌다 나타나면 추적 ID가 바뀌어 새 사람으로 집계될 수 있다.
- 수업에서 쓸 때는 참가자에게 촬영 사실을 알리고, 동작 기록도 수업 목적으로만 쓴다.
- 확장 과제: "손들기(질문)" 감지 시 8단계 웹에 알림을 띄우거나, 만세 횟수를 팀 투표에 활용해 본다.

## 10단계 USB 카메라 2대로 특정 물체 접근 감지

강의실의 특정 물체(타겟)에 사람이 접근하는지 판단하고, 정해 둔 거리보다 가까워지거나 멀어지면 알람을 울린다. 교실에서는 책상에 발이 가려지므로, 책상 위로 보이는 어깨 중심을 두 카메라로 삼각측량해 바닥 위치를 구하는 것이 핵심이다. 실습은 **보정(10-1, 10-2) → 문제 확인(10-3) → 개선(10-4) → 보완(10-5) → 접근 알람(10-6)** 순서로 진행한다.

| 역할 | 담당 |
| --- | --- |
| 타겟 찾기 | YOLOE가 텍스트(예: "red fire extinguisher")로 후보 박스를 내고, EmbeddingGemma 2가 크롭과 문구를 비교해 가장 맞는 박스를 고름. 두 카메라 박스 중심을 삼각측량 |
| 사람 위치 | YOLO11n-pose의 어깨 중심(없으면 코) → 두 카메라 삼각측량 → 바닥 위 (x, y)와 높이 |
| 같은 사람 맞추기 | 얼굴·외모는 쓰지 않고, 재투영 오차가 작은 쌍끼리 매칭 |
| 접근 판단·알람 | 최근 1초 거리 변화로 접근/이탈, 근접·이탈 경계는 히스테리시스(0.2m)로 깜빡임 방지 |

**준비물**

- USB 웹캠 2대 (640x480 MJPG 지원), 필요하면 USB 연장선 (5m 이내)
- 체커보드 인쇄물 (내부 코너 9x6, 칸 25mm, 두꺼운 판에 붙여 평평하게)
- 줄자, 마스킹 테이프 (기준점 표시용), 타겟 물체

**카메라 배치**

- 강의실 앞쪽 두 모서리 부근, 높이 2.2\~2.5m에 고정하고 둘 다 교실 중앙(타겟 방향)을 향하게 한다.
- 타겟 주변에서 두 시선이 60\~90도로 만나면 가장 좋다. 나란히 붙이면 거리 방향 오차가 커지고, 마주 보면 매칭이 자주 실패한다.
- 보정을 끝낸 뒤에는 카메라를 절대 움직이지 않는다. 움직이면 10-2를 다시 한다.
- 웹캠은 하나가 `/dev/video` 두 개를 만드는 경우가 많다. `v4l2-ctl --list-devices`로 실제 영상 장치 번호(예: 0과 2)를 확인하고, 이후 모든 명령에 같은 순서로 쓴다.

**파일 구성** (`~/eg2-lab/prox/`)

| 파일 | 용도 |
| --- | --- |
| `stereo.py` | 공통: 보정값 로드, 왜곡 보정, 삼각측량, 단일 카메라 높이 평면 추정, 사람 매칭 |
| `calib_intrinsics.py` | 10-1 카메라별 렌즈 보정 (체커보드) |
| `calib/world_points.json` | 10-2 기준점 실제 좌표 (직접 측정해 수정) |
| `calib_extrinsics.py` | 10-2 웹에서 기준점을 클릭해 카메라 위치·방향 계산 |
| `proximity_app.py` | 10-3\~10-6 접근 감지 웹 앱 (`--mode ankle1 / stereo / fused`) |

좌표계는 강의실 바닥 기준이다. 원점은 칠판 왼쪽 아래 바닥, x는 칠판과 평행(오른쪽 +), y는 교실 뒤쪽(+), z는 높이(위 +), 단위는 m다.

**공통 모듈** `stereo.py`:

```python
"""10단계 공통: 카메라 보정값 로드, 왜곡 보정, 삼각측량, 단일 카메라 높이 평면 추정
좌표계: 강의실 바닥 기준. x = 칠판과 평행(오른쪽 +), y = 교실 뒤쪽(+), z = 높이(위 +), 단위 m
"""
import json
import numpy as np
import cv2

KP_CONF = 0.5
NOSE, LS, RS, LA, RA = 0, 5, 6, 15, 16

def load_cam(path):
    d = np.load(path)
    K, dist, R, t = d["K"], d["dist"], d["R"], d["t"].reshape(3, 1)
    return {"K": K, "dist": dist, "R": R, "t": t, "P": K @ np.hstack([R, t]),
            "C": (-R.T @ t).ravel(), "Kinv": np.linalg.inv(K)}

def load_world(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["points"]

def room_bounds(points, pad=0.5):
    xy = np.array([p["xyz"][:2] for p in points], dtype=float)
    return [float(xy[:, 0].min() - pad), float(xy[:, 1].min() - pad),
            float(xy[:, 0].max() + pad), float(xy[:, 1].max() + pad)]

def undistort(cam, pts):
    """왜곡이 있는 픽셀 좌표 → 이상적인(왜곡 없는) 픽셀 좌표 (N, 2)"""
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    if len(pts) == 0:
        return np.zeros((0, 2))
    return cv2.undistortPoints(pts, cam["K"], cam["dist"], P=cam["K"]).reshape(-1, 2)

def project(cam, X):
    """3D 점 → 왜곡 없는 픽셀 좌표"""
    x = cam["P"] @ np.append(np.asarray(X, float), 1.0)
    return x[:2] / x[2]

def project_distorted(cam, X):
    """3D 점 → 실제 영상 픽셀 좌표 (화면 표시용)"""
    rvec, _ = cv2.Rodrigues(cam["R"])
    p, _ = cv2.projectPoints(np.float64([X]), rvec, cam["t"], cam["K"], cam["dist"])
    return p.reshape(2)

def triangulate(c0, c1, p0, p1):
    """두 카메라의 왜곡 없는 픽셀 좌표 → 3D 점, 평균 재투영 오차(px)"""
    X = cv2.triangulatePoints(c0["P"], c1["P"], np.float64(p0).reshape(2, 1), np.float64(p1).reshape(2, 1))
    X = (X[:3] / X[3]).ravel()
    err = (np.linalg.norm(project(c0, X) - p0) + np.linalg.norm(project(c1, X) - p1)) / 2
    return X, float(err)

def ray_plane(cam, p, h):
    """카메라 1대: 픽셀(왜곡 없음)을 지나는 시선이 높이 h 평면과 만나는 점"""
    d = cam["R"].T @ (cam["Kinv"] @ np.array([p[0], p[1], 1.0]))
    C = cam["C"]
    if abs(d[2]) < 1e-6:
        return None
    s = (h - C[2]) / d[2]
    return C + s * d if s > 0 else None

def upper_body(k):
    """어깨 중심, 어깨가 안 보이면 코. 책상에 가려지지 않는 관절"""
    if k[LS, 2] > KP_CONF and k[RS, 2] > KP_CONF:
        return (k[LS, :2] + k[RS, :2]) / 2
    if k[NOSE, 2] > KP_CONF:
        return k[NOSE, :2].copy()
    return None

def ankle(k):
    """보이는 발목의 평균 (10-3 문제 확인용)"""
    vis = [k[i, :2] for i in (LA, RA) if k[i, 2] > KP_CONF]
    return np.mean(vis, axis=0) if vis else None

def match_people(c0, c1, pts0, pts1, max_err=15.0, h_range=(0.5, 2.1)):
    """두 카메라의 사람(왜곡 없는 상체 좌표)을 재투영 오차가 작은 순으로 1:1 매칭
    반환: [(i, j, X(3,), err)], 매칭 안 된 cam0 인덱스 집합, cam1 인덱스 집합"""
    cands = []
    for i, a in enumerate(pts0):
        for j, b in enumerate(pts1):
            X, err = triangulate(c0, c1, a, b)
            if err <= max_err and h_range[0] <= X[2] <= h_range[1]:
                cands.append((err, i, j, X))
    cands.sort(key=lambda c: c[0])
    used0, used1, out = set(), set(), []
    for err, i, j, X in cands:
        if i in used0 or j in used1:
            continue
        out.append((i, j, X, err))
        used0.add(i); used1.add(j)
    return out, set(range(len(pts0))) - used0, set(range(len(pts1))) - used1

# ---------------- 10-8 이동 타겟: 칼만 필터와 상대 운동 ----------------
class Kalman2D:
    """등속 모델 칼만 필터. 상태 (x, y, vx, vy), 측정 (x, y). q: 가속 잡음, r: 측정 잡음(m²)"""
    def __init__(self, xy, q=1.0, r=0.05):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0], float)
        self.P = np.diag([r, r, 1.0, 1.0])
        self.q, self.r = q, r

    def predict(self, dt):
        if dt <= 0:
            return
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.q * G @ G.T

    def update(self, z):
        Hm = np.eye(2, 4)
        S_ = Hm @ self.P @ Hm.T + self.r * np.eye(2)
        K = self.P @ Hm.T @ np.linalg.inv(S_)
        self.x = self.x + K @ (np.asarray(z, float) - Hm @ self.x)
        self.P = (np.eye(4) - K @ Hm) @ self.P

    def at(self, dt):
        """dt초 뒤 예측 위치 (상태는 바꾸지 않음)"""
        return self.x[:2] + self.x[2:] * dt

def relative_state(p, t):
    """p, t: 상태 벡터 (x, y, vx, vy). 반환: 거리(m), 접근 속도(m/s, +면 가까워짐), TTC(초 또는 None)"""
    dp, dv = t[:2] - p[:2], t[2:] - p[2:]
    d = float(np.linalg.norm(dp))
    closing = -float(dp @ dv) / max(d, 1e-6)
    ttc = d / closing if closing > 0.1 else None
    return d, closing, ttc
```

### 10-1 렌즈 보정 (카메라별 1회)

웹캠 렌즈는 가장자리가 휘어 보이는 왜곡이 있어 그대로 삼각측량하면 오차가 커진다. 체커보드를 화면 고루고루, 여러 각도로 비추면 20장을 자동으로 모아 내부 행렬(K)과 왜곡 계수를 저장한다. 화면(모니터) 없이 동작한다.

`calib_intrinsics.py`:

```python
"""10-1 렌즈 보정: 체커보드를 여러 각도로 비추면 자동으로 20장을 모아 K(내부 행렬), dist(왜곡)를 저장
화면이 없어도 동작한다 (진행 상황은 터미널, 마지막 검출 이미지는 calib/camN_last.jpg)
사용법: python calib_intrinsics.py --cam 0
"""
import argparse, os, time
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, required=True)
ap.add_argument("--cols", type=int, default=9, help="체커보드 내부 코너 가로 개수")
ap.add_argument("--rows", type=int, default=6, help="체커보드 내부 코너 세로 개수")
ap.add_argument("--square", type=float, default=0.025, help="칸 한 변 길이(m)")
ap.add_argument("--n", type=int, default=20, help="모을 이미지 수")
ap.add_argument("--res", default="640x480")
args = ap.parse_args()
W, H = map(int, args.res.split("x"))
os.makedirs("calib", exist_ok=True)

objp = np.zeros((args.rows * args.cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square

cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
assert cap.isOpened(), f"카메라 {args.cam}을 열 수 없음"

obj_pts, img_pts, last_t, last_c = [], [], 0.0, None
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
print(f"체커보드({args.cols}x{args.rows})를 화면 곳곳·여러 각도로 천천히 움직이세요. 목표 {args.n}장")
while len(img_pts) < args.n:
    ok, frame = cap.read()
    if not ok:
        continue
    assert frame.shape[1] == W, f"해상도가 {W}x{H}가 아님: {frame.shape[1]}x{frame.shape[0]}"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows), cv2.CALIB_CB_FAST_CHECK)
    if not found or time.time() - last_t < 1.0:
        continue
    c = corners.reshape(-1, 2).mean(0)
    if last_c is not None and np.linalg.norm(c - last_c) < 40:     # 비슷한 위치는 건너뜀
        continue
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    obj_pts.append(objp); img_pts.append(corners)
    last_t, last_c = time.time(), c
    cv2.drawChessboardCorners(frame, (args.cols, args.rows), corners, True)
    cv2.imwrite(f"calib/cam{args.cam}_last.jpg", frame)
    print(f"  {len(img_pts)}/{args.n}")

rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, (W, H), None, None)
np.savez(f"calib/cam{args.cam}_intr.npz", K=K, dist=dist, res=np.array([W, H]))
print(f"\n재투영 RMS {rms:.3f}px  (0.5 이하 양호, 1.0 이상이면 다시 촬영)")
print(f"K=\n{K.round(1)}\ndist={dist.ravel().round(3)}\n저장: calib/cam{args.cam}_intr.npz")
```

1. 카메라 장치 번호를 확인한다.
2. 카메라마다 한 번씩 실행하고, 체커보드를 화면 가장자리와 모서리까지 기울여 가며 천천히 움직인다. 설치 전 책상 위에서 해도 된다.
3. 끝에 출력되는 RMS가 0.5px 이하인지 확인하고 아래 표에 적는다. 1.0 이상이면 다시 촬영한다.

```bash
mkdir -p ~/eg2-lab/prox/calib && cd ~/eg2-lab/prox && source ~/eg2/bin/activate
v4l2-ctl --list-devices          # 예: /dev/video0, /dev/video2
python calib_intrinsics.py --cam 0
python calib_intrinsics.py --cam 2
ls calib/                        # cam0_intr.npz, cam2_intr.npz
```

| 카메라 | 장치 번호 | RMS (px) |
| --- | --- | --- |
| cam0 |  |  |
| cam1 |  |  |

### 10-2 카메라 위치 보정 (설치 후 1회)

강의실에 기준점을 표시하고 실제 좌표를 잰 뒤, 웹 화면에서 두 카메라 스냅샷 위의 같은 점을 클릭하면 각 카메라의 위치와 방향을 계산한다(`solvePnP`). 체커보드를 이용한 스테레오 보정보다 간단하고, 강의실 좌표계가 바로 정해진다.

기준점 고르는 법:

- 두 카메라 모두에 보이는 점 8개 이상. 계산용 6개 이상 + 검증용(`"check": true`) 2개 이상
- 높이가 다른 점을 섞는다: 바닥 테이프(z=0), 책상 모서리(z≈0.74), 벽면 표시(z≈1.5). 모두 바닥에만 있으면 계산이 불안정해진다
- 교실 전체에 고루 퍼지게, 특히 타겟 주변에 2\~3개

`calib/world_points.json` (예시. 실제 측정값으로 바꾼다):

```json
{
  "_설명": "강의실 기준점의 실제 좌표(m). 원점=칠판 왼쪽 아래 바닥, x=칠판 방향(오른쪽 +), y=교실 뒤쪽(+), z=높이. 두 카메라 모두에 보이는 점, 높이가 서로 다른 점(바닥·책상·벽)을 섞는다. check=true 는 계산에 쓰지 않는 검증용",
  "points": [
    {"id": "F1", "xyz": [1.0, 1.0, 0.0]},
    {"id": "F2", "xyz": [5.0, 1.0, 0.0]},
    {"id": "F3", "xyz": [5.0, 5.0, 0.0]},
    {"id": "F4", "xyz": [1.0, 5.0, 0.0]},
    {"id": "D1", "xyz": [2.0, 3.0, 0.74]},
    {"id": "D2", "xyz": [4.0, 3.0, 0.74]},
    {"id": "W1", "xyz": [0.0, 3.0, 1.5]},
    {"id": "W2", "xyz": [6.0, 3.0, 1.5]},
    {"id": "C1", "xyz": [3.0, 2.0, 0.0], "check": true},
    {"id": "C2", "xyz": [3.0, 4.0, 0.74], "check": true}
  ]
}
```

`calib_extrinsics.py`:

```python
"""10-2 카메라 위치 보정: 웹에서 두 카메라 스냅샷 위의 기준점을 클릭 → solvePnP로 카메라 위치·방향 계산
기준점 실제 좌표는 calib/world_points.json에 미리 적어 둔다 ("check": true 인 점은 계산에 쓰지 않고 검증에만 사용)
사용법: python calib_extrinsics.py --cams 0 1   →  브라우저에서 http://<Jetson IP>:8000
"""
import argparse, json, os, time
import cv2, numpy as np, uvicorn
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, Response
from stereo import load_world, undistort, triangulate, load_cam

ap = argparse.ArgumentParser()
ap.add_argument("--cams", type=int, nargs=2, default=[0, 1])
ap.add_argument("--res", default="640x480")
ap.add_argument("--port", type=int, default=8000)
args = ap.parse_args()
W, H = map(int, args.res.split("x"))
WORLD = load_world("calib/world_points.json")
WMAP = {p["id"]: p for p in WORLD}
CLICKS_F = "calib/clicks.json"
clicks = json.load(open(CLICKS_F)) if os.path.exists(CLICKS_F) else {"0": {}, "1": {}}
snaps = {}

def open_cam(i):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    assert cap.isOpened(), f"카메라 {i}을 열 수 없음"
    return cap

def take_snaps():
    for n, ci in enumerate(args.cams):
        cap = open_cam(ci)
        for _ in range(15):          # 자동 노출 안정화
            ok, f = cap.read()
        cap.release()
        snaps[str(n)] = f
        cv2.imwrite(f"calib/snap{n}.jpg", f)

def intr(n):
    d = np.load(f"calib/cam{args.cams[n]}_intr.npz")
    return d["K"], d["dist"]

def solve():
    out = {}
    for n in ("0", "1"):
        K, dist = intr(int(n))
        ids = [i for i in clicks[n] if i in WMAP and not WMAP[i].get("check")]
        if len(ids) < 6:
            out[n] = {"error": f"계산용 기준점 {len(ids)}개 (6개 이상 필요)"}; continue
        obj = np.float64([WMAP[i]["xyz"] for i in ids])
        img = np.float64([clicks[n][i] for i in ids])
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        errs = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
        R, _ = cv2.Rodrigues(rvec)
        np.savez(f"calib/cam{n}.npz", K=K, dist=dist, R=R, t=tvec, res=np.array([W, H]))
        C = (-R.T @ tvec).ravel()
        out[n] = {"pos": C.round(2).tolist(), "reproj": {i: round(float(e), 1) for i, e in zip(ids, errs)},
                  "mean_px": round(float(errs.mean()), 2)}
    # 두 카메라 모두 클릭한 점으로 3D 검증 (check 점 포함)
    if "error" not in out.get("0", {"error": 1}) and "error" not in out.get("1", {"error": 1}):
        c0, c1 = load_cam("calib/cam0.npz"), load_cam("calib/cam1.npz")
        ver = {}
        for i in set(clicks["0"]) & set(clicks["1"]) & set(WMAP):
            p0, p1 = undistort(c0, [clicks["0"][i]])[0], undistort(c1, [clicks["1"][i]])[0]
            X, _ = triangulate(c0, c1, p0, p1)
            ver[i] = {"check": bool(WMAP[i].get("check")),
                      "err_cm": round(float(np.linalg.norm(X - np.array(WMAP[i]["xyz"]))) * 100, 1)}
        out["verify"] = ver
    return out

app = FastAPI()

@app.get("/img/{n}.jpg")
def img(n: str):
    f = snaps[n].copy()
    for i, (u, v) in clicks[n].items():
        col = (0, 200, 255) if WMAP.get(i, {}).get("check") else (0, 255, 0)
        cv2.circle(f, (int(u), int(v)), 5, col, -1)
        cv2.putText(f, i, (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return Response(cv2.imencode(".jpg", f)[1].tobytes(), media_type="image/jpeg")

@app.post("/click")
def click(d: dict = Body(...)):
    clicks[str(d["cam"])][d["id"]] = [float(d["u"]), float(d["v"])]
    json.dump(clicks, open(CLICKS_F, "w"), indent=1)
    return {"ok": True}

@app.post("/undo")
def undo(d: dict = Body(...)):
    clicks[str(d["cam"])].pop(d["id"], None)
    json.dump(clicks, open(CLICKS_F, "w"), indent=1)
    return {"ok": True}

@app.post("/snap")
def snap():
    take_snaps(); return {"ok": True}

@app.post("/solve")
def do_solve():
    return solve()

@app.get("/points")
def points():
    return {"points": WORLD, "clicks": clicks}

PAGE = """<!doctype html><meta charset=utf-8><title>카메라 위치 보정</title>
<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}.row{display:flex;gap:12px;flex-wrap:wrap}
.row div{flex:1;min-width:300px}img{width:100%;cursor:crosshair;border:1px solid #444}select,button{font-size:15px;padding:4px 10px}
pre{background:#222;padding:8px;white-space:pre-wrap}</style>
<h2>카메라 위치 보정</h2>
<p>기준점 선택 → 해당 카메라 이미지에서 그 점을 클릭. 초록 = 계산용, 주황 = 검증용(check). 카메라마다 계산용 6개 이상.</p>
<p><select id=pid></select> <button onclick="undo(0)">cam0 선택점 지우기</button> <button onclick="undo(1)">cam1 선택점 지우기</button>
<button onclick="snapNew()">새 스냅샷</button> <button onclick="solve()">계산</button></p>
<div class=row><div>cam0<br><img id=i0 src="/img/0.jpg" onclick="clk(event,0)"></div>
<div>cam1<br><img id=i1 src="/img/1.jpg" onclick="clk(event,1)"></div></div>
<h3>결과</h3><pre id=out>-</pre>
<script>
async function load(){const r=await (await fetch('/points')).json();
 document.getElementById('pid').innerHTML=r.points.map(p=>`<option value=${p.id}>${p.id} (${p.xyz.join(', ')})${p.check?' 검증':''}</option>`).join('');}
function refresh(){for(const n of [0,1])document.getElementById('i'+n).src=`/img/${n}.jpg?`+Date.now();}
async function clk(e,n){const im=e.target,u=e.offsetX*im.naturalWidth/im.clientWidth,v=e.offsetY*im.naturalHeight/im.clientHeight;
 await fetch('/click',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cam:n,id:pid.value,u,v})});refresh();}
async function undo(n){await fetch('/undo',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cam:n,id:pid.value})});refresh();}
async function snapNew(){await fetch('/snap',{method:'POST'});refresh();}
async function solve(){const r=await (await fetch('/solve',{method:'POST'})).json();document.getElementById('out').textContent=JSON.stringify(r,null,1);}
load();
</script>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE

if __name__ == "__main__":
    take_snaps()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
```

1. 기준점에 테이프를 붙이고 줄자로 좌표를 재서 `calib/world_points.json`을 수정한다. 스냅샷에서 잘 보이도록 테이프는 십자(+) 모양으로 붙이고, 교차점을 기준으로 잰다.
2. `--cams`에 10-1의 장치 번호를 같은 순서로 넣어 실행하고, 브라우저로 접속한다. 스냅샷에 사람이 기준점을 가리지 않게 한다.
3. 드롭다운에서 기준점을 고르고, cam0·cam1 이미지에서 각각 그 점을 클릭한다. 모든 기준점을 반복한다. 클릭은 `calib/clicks.json`에 저장돼 다시 실행해도 유지된다.
4. **계산**을 누른다. 카메라별 평균 재투영 오차(`mean_px`), 계산된 카메라 위치(`pos`), 기준점별 3D 오차(`verify`, cm)가 나온다.
5. `pos`가 실제 카메라 설치 위치와 대략 맞는지 줄자로 확인한다. 특정 점의 오차만 크면 클릭을 지우고 다시 찍거나, 좌표 측정을 다시 확인한다.

```bash
cd ~/eg2-lab/prox && source ~/eg2/bin/activate
nano calib/world_points.json      # 실제 측정 좌표로 수정
python calib_extrinsics.py --cams 0 2
ls calib/                         # cam0.npz, cam1.npz 생성 확인
```

| 항목 | 목표 | cam0 | cam1 |
| --- | --- | --- | --- |
| 평균 재투영 오차 (px) | 3 이하 |  |  |
| 계산 위치 vs 실측 위치 (cm) | 20 이내 |  |  |
| 검증점 C1 3D 오차 (cm) | 10 이내 | (공통) |  |
| 검증점 C2 3D 오차 (cm) | 10 이내 | (공통) |  |

검증점 오차가 이 실습에서 기대할 수 있는 위치 정확도다. 두 카메라를 가상으로 배치한 시뮬레이션(관절 오차 2px)에서는 삼각측량 위치 오차가 5cm 안팎이었다. 실제로는 클릭·측정 오차가 더해져 10cm 안팎이면 양호하다.

### 접근 감지 앱 코드

`proximity_app.py` 하나로 10-3\~10-6을 모두 실행한다. `--mode`로 위치 계산 방식을 바꾼다.

| `--mode` | 카메라 | 사람 위치 | 단계 |
| --- | --- | --- | --- |
| `ankle1` | 1대 | 발목 → 바닥(z=0) 평면 | 10-3 문제 확인 |
| `stereo` | 2대 | 어깨 중심 삼각측량 | 10-4 개선 |
| `fused` (기본) | 2대 | 삼각측량 + 한 카메라에만 보이면 높이 1.2m 평면으로 추정 | 10-5 보완 |

```python
"""10단계: USB 카메라 2대로 강의실 특정 물체(타겟)에 대한 사람의 접근 감지·알람
--mode ankle1 : 카메라 1대 + 발목 → 바닥(z=0) 위치 (10-3 문제 확인)
--mode stereo : 카메라 2대 + 상체(어깨 중심) 삼각측량 (10-4 개선)
--mode fused  : stereo + 한 카메라에만 보이는 사람은 높이 평면으로 추정 (10-5 보완, 기본)
타겟 위치: --target "문구"로 YOLOE 후보 + EmbeddingGemma 2 재확인 후 삼각측량, 또는 --target-xy x,y 직접 지정
--moving      : 타겟도 매 프레임 추적 (칼만 필터 + 상대 속도 + 충돌 예상 시간 TTC 알람) (10-8)
사용법: python proximity_app.py --cams 0 2 --target "red fire extinguisher"
        python proximity_app.py --cams 0 2 --target "cart" --moving
"""
import argparse, time, threading, collections
import cv2, numpy as np, uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO
import stereo as S

ap = argparse.ArgumentParser()
ap.add_argument("--cams", type=int, nargs=2, default=[0, 1], help="calib_extrinsics.py와 같은 순서")
ap.add_argument("--mode", choices=["ankle1", "stereo", "fused"], default="fused")
ap.add_argument("--target", default=None, help='예: "red fire extinguisher"')
ap.add_argument("--target-xy", default=None, help="예: 3.0,1.2 (직접 측정한 바닥 좌표)")
ap.add_argument("--target-h", type=float, default=0.8, help="한 카메라만 타겟을 볼 때 가정할 높이(m)")
ap.add_argument("--near", type=float, default=1.0, help="근접 알람 거리(m)")
ap.add_argument("--far", type=float, default=3.0, help="이탈 알람 거리(m)")
ap.add_argument("--moving", action="store_true", help="이동 타겟 모드 (10-8)")
ap.add_argument("--ttc", type=float, default=2.0, help="충돌 위험 알람 TTC 기준(초)")
ap.add_argument("--target-every", type=int, default=2, help="이동 타겟 검출 주기(프레임)")
ap.add_argument("--res", default="640x480")
ap.add_argument("--port", type=int, default=8000)
args = ap.parse_args()
W, H = map(int, args.res.split("x"))
if args.moving and not args.target:
    ap.error("--moving 에는 --target \"문구\"가 필요합니다 (매 프레임 YOLOE로 타겟을 찾기 때문)")

MAX_SKEW = 0.08       # 두 카메라 프레임 시간 차 허용(초). 넘으면 그 순간은 삼각측량 생략
H_GUESS = 1.2         # 한 카메라 추정 시 어깨 높이 가정(m): 앉음 약 1.0, 섬 약 1.4의 중간
GATE = 0.8            # 같은 사람으로 이어 붙일 최대 이동(m/프레임)
SMOOTH_S = 0.5        # 위치 중앙값 창(초)
SIT_H = 1.15          # 이 높이 미만이면 '앉음'
NUM_CAMS = 1 if args.mode == "ankle1" else 2
CLOSE_V = 0.5         # 충돌 위험 알람: 접근 속도(m/s) 기준
TGT_LOST_S = 2.0      # 이 시간 이상 타겟을 못 보면 EmbeddingGemma로 다시 식별

cams = [S.load_cam(f"calib/cam{n}.npz") for n in range(NUM_CAMS)]
ROOM = S.room_bounds(S.load_world("calib/world_points.json"))
det = YOLO("yolo11n-pose.pt")

# ---------------- 카메라 수신 (최신 프레임만 보관) ----------------
class Cam(threading.Thread):
    def __init__(self, dev):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        assert self.cap.isOpened(), f"카메라 {dev}을 열 수 없음 (Q11, Q14 참고)"
        self.lock, self.frame, self.ts = threading.Lock(), None, 0.0
    def run(self):
        while True:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame, self.ts = f, time.time()
    def get(self):
        with self.lock:
            return self.ts, (None if self.frame is None else self.frame.copy())

# ---------------- 접근·이탈 알람 ----------------
class ProximityAlarm:
    """히스테리시스로 경계에서 깜빡이지 않게 하고, 최근 1초 거리 변화 속도로 접근/이탈 판단"""
    def __init__(self, near, far, margin=0.2, v_th=0.3):
        self.near, self.far, self.m, self.v_th = near, far, margin, v_th
        self.zone, self.hist, self.trend, self.ttc_on = "mid", [], "정지", False
    def update_ttc(self, closing, ttc, d):
        """이동 타겟: 빠르게 가까워지고 TTC가 짧으면 충돌 위험 알람 (조건이 풀리면 다시 무장)"""
        danger = closing > CLOSE_V and ttc is not None and ttc < args.ttc
        ev = []
        if danger and not self.ttc_on:
            ev.append(f"알람: 충돌 위험 TTC {ttc:.1f}초 (접근 {closing:.1f}m/s, {d:.1f}m)")
        if danger:
            self.ttc_on = True
        elif self.ttc_on and (closing < 0.2 or ttc is None or ttc > args.ttc * 1.5):
            self.ttc_on = False
        return ev
    def update(self, t, d):
        self.hist = [(tt, dd) for tt, dd in self.hist if t - tt < 1.0] + [(t, d)]
        span = self.hist[-1][0] - self.hist[0][0]
        v = (self.hist[-1][1] - self.hist[0][1]) / span if span > 0.3 else 0.0
        trend = "접근 중" if v < -self.v_th else "멀어지는 중" if v > self.v_th else "정지"
        ev = []
        if trend == "접근 중" and self.trend != "접근 중" and d < self.far:
            ev.append(f"접근 시작 ({d:.1f}m)")
        self.trend = trend
        if self.zone != "near" and d < self.near:
            self.zone = "near"; ev.append(f"알람: 근접 {d:.1f}m")
        elif self.zone == "near" and d > self.near + self.m:
            self.zone = "mid"
        if self.zone != "far" and d > self.far:
            if self.zone == "mid" and self.hist[0][1] <= self.far:   # 안쪽에 있다가 나간 경우만 알람
                ev.append(f"알람: 이탈 {d:.1f}m")
            self.zone = "far"
        elif self.zone == "far" and d < self.far - self.m:
            self.zone = "mid"
        return ev

# ---------------- 바닥 평면 추적 ----------------
class Track:
    _next = 1
    def __init__(self, xy, h, kind, now):
        self.id, Track._next = Track._next, Track._next + 1
        self.hist = collections.deque([(now, xy, h)], maxlen=60)
        self.kind, self.last, self.first = kind, now, now
        self.alarm = ProximityAlarm(args.near, args.far)
        self.kf = S.Kalman2D(xy)
        self.rel = (None, 0.0, None)          # (거리, 접근 속도, TTC)
    def observe(self, xy, h, kind, now):
        self.kf.predict(now - self.last); self.kf.update(xy)
        self.hist.append((now, xy, h)); self.kind, self.last = kind, now
    @property
    def xy(self):
        if args.moving:                       # 이동 모드: 칼만 필터 위치 (속도 추정 포함)
            return self.kf.x[:2].copy()
        pts = [p for t, p, _ in self.hist if self.last - t <= SMOOTH_S]
        return np.median(pts, axis=0)
    @property
    def h(self):
        return float(np.median([hh for t, _, hh in self.hist if self.last - t <= SMOOTH_S]))

tracks, events = [], collections.deque(maxlen=60)
ev_lock = threading.Lock()
state = {"target": None, "stats": {}, "people": []}
tgt = {"kf": None, "h": args.target_h, "last": 0.0, "boxes": [None, None], "how": "",
       "trail": collections.deque(maxlen=50), "ref": None}
models = {}           # 이동 모드에서 계속 쓰는 yoloe / emb / qv
jpegs = [None, None]

def log(now, who, msg):
    with ev_lock:
        events.appendleft({"ts": now, "who": who, "label": msg})

def update_tracks(obs, now):
    """obs: [(xy, h, kind)] → 가장 가까운 트랙에 이어 붙이고(GATE 이내), 나머지는 새 트랙"""
    free = list(range(len(tracks)))
    for xy, h, kind in sorted(obs, key=lambda o: o[2] != "stereo"):     # 삼각측량 결과 우선
        best, bi = GATE, None
        for i in free:
            d = np.linalg.norm(tracks[i].xy - xy)
            if d < best:
                best, bi = d, i
        if bi is None:
            tracks.append(Track(xy, h, kind, now)); log(now, f"#{tracks[-1].id}", "등장")
        else:
            tr = tracks[bi]; tr.observe(xy, h, kind, now)
            free.remove(bi)
    for tr in [t for t in tracks if now - t.last > 2.0]:
        tracks.remove(tr); log(now, f"#{tr.id}", f"퇴장 (머문 시간 {tr.last - tr.first:.0f}초)")

# ---------------- 타겟 찾기: YOLOE 후보 → EmbeddingGemma 2 재확인 → 삼각측량 ----------------
def load_target_models():
    import torch
    from ultralytics import YOLOE
    from sentence_transformers import SentenceTransformer
    if not models:
        yoloe = YOLOE("yoloe-11s-seg.pt")
        yoloe.set_classes([args.target], yoloe.get_text_pe([args.target]))
        emb = SentenceTransformer("google/embeddinggemma-2", device="cuda",
                                  model_kwargs={"torch_dtype": torch.bfloat16}, config_kwargs={"audio_config": None})
        qv = np.asarray(emb.encode(f"a photo of a {args.target}", prompt_name="SearchQuery",
                                   normalize_embeddings=True), dtype=np.float32)
        models.update(yoloe=yoloe, emb=emb, qv=qv)
    return models["yoloe"], models["emb"], models["qv"]

def embed_crop(emb, f, box, path):
    x1, y1, x2, y2 = box
    cv2.imwrite(path, f[y1:y2, x1:x2])
    return np.asarray(emb.encode({"image": path}, normalize_embeddings=True), dtype=np.float32)

def find_target(frames):
    import torch
    yoloe, emb, qv = load_target_models()
    found = []
    for n, f in enumerate(frames):
        r = yoloe.predict(f, conf=0.1, verbose=False)[0]
        best = None
        for i, (x1, y1, x2, y2) in enumerate(r.boxes.xyxy.int().tolist()):
            if min(x2 - x1, y2 - y1) < 16:
                continue
            v = embed_crop(emb, f, (x1, y1, x2, y2), f"calib/target_cand{n}_{i}.jpg")
            score = float(v @ qv)
            if best is None or score > best[0]:
                best = (score, ((x1 + x2) / 2, (y1 + y2) / 2), (x1, y1, x2, y2), v)
        found.append(best)
    ok = [b is not None and b[0] >= 0.15 for b in found]
    if any(ok):                                    # 이동 모드 재식별용 기준 임베딩 (점수가 높은 쪽 크롭)
        tgt["ref"] = max((b for b, o in zip(found, ok) if o), key=lambda b: b[0])[3]
    if not args.moving:
        models.clear()
        torch.cuda.empty_cache()                   # 고정 타겟은 한 번 찾고 메모리 반환
    if NUM_CAMS == 2 and all(ok):
        p0, p1 = S.undistort(cams[0], [found[0][1]])[0], S.undistort(cams[1], [found[1][1]])[0]
        X, err = S.triangulate(cams[0], cams[1], p0, p1)
        how = f"삼각측량 (재투영 {err:.1f}px)"
    elif any(ok):
        n = ok.index(True)
        X = S.ray_plane(cams[n], S.undistort(cams[n], [found[n][1]])[0], args.target_h)
        how = f"cam{n} 단독 추정 (높이 {args.target_h}m 가정)"
    else:
        return None
    return {"xyz": X.round(2).tolist(), "how": how,
            "boxes": [b[2] if o else None for b, o in zip(found, ok)],
            "scores": [round(b[0], 3) if b else None for b in found]}

def set_target(frames):
    if args.target_xy:
        x, y = map(float, args.target_xy.split(","))
        state["target"] = {"xyz": [x, y, 0.0], "how": "직접 지정", "boxes": [None, None], "scores": []}
    elif args.target:
        t = find_target(frames)
        state["target"] = t
        log(time.time(), "타겟", f"찾음: {t['xyz']} / {t['how']} / 점수 {t['scores']}" if t else "찾지 못함")
    if args.moving and state["target"]:
        x, y, z = state["target"]["xyz"]
        tgt.update(kf=S.Kalman2D((x, y)), last=time.time(), boxes=state["target"]["boxes"],
                   how=state["target"]["how"])
        if z > 0.1:
            tgt["h"] = z
        tgt["trail"].clear()
    retarget.clear()

def track_target(frames, now, idx):
    """10-8: 이동 타겟을 매 N프레임 검출 → 예측 위치에 가까운 후보 선택(놓쳤으면 임베딩으로 재식별) → 삼각측량 → 칼만 갱신"""
    kf = tgt["kf"]
    if kf is None:
        return
    if idx % args.target_every == 0:
        yoloe, emb, qv = load_target_models()
        lost = now - tgt["last"] > TGT_LOST_S
        X_pred = [*kf.at(now - tgt["last"]), tgt["h"]]
        chosen = [None, None]
        for n, f in enumerate(frames):
            r = yoloe.predict(f, conf=0.1, verbose=False)[0]
            boxes = [b for b in r.boxes.xyxy.int().tolist() if min(b[2] - b[0], b[3] - b[1]) >= 16]
            if not boxes:
                continue
            centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
            if not lost:                       # 추적 중: 예측 위치를 화면에 투영해 가장 가까운 후보 (100px 이내)
                pp = S.project_distorted(cams[n], X_pred)
                dists = [np.hypot(c[0] - pp[0], c[1] - pp[1]) for c in centers]
                i = int(np.argmin(dists))
                if dists[i] < 100:
                    chosen[n] = (centers[i], boxes[i])
            else:                              # 놓침: 기준 임베딩과 가장 닮은 후보 (큰 박스 5개까지)
                ref = tgt["ref"] if tgt["ref"] is not None else qv
                cand = sorted(range(len(boxes)), key=lambda i: -(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))[:5]
                sc = [(float(embed_crop(emb, f, boxes[i], f"calib/reid{n}_{i}.jpg") @ ref), i) for i in cand]
                s_best, i = max(sc)
                if s_best >= 0.15:
                    chosen[n] = (centers[i], boxes[i])
        X, how = None, ""
        if NUM_CAMS == 2 and all(chosen):
            p0, p1 = S.undistort(cams[0], [chosen[0][0]])[0], S.undistort(cams[1], [chosen[1][0]])[0]
            Xt, err = S.triangulate(cams[0], cams[1], p0, p1)
            if err < 15:
                X, how = Xt, "삼각측량"
                tgt["h"] = 0.8 * tgt["h"] + 0.2 * float(Xt[2])     # 타겟 높이는 천천히 갱신
        if X is None and any(chosen):
            n = 0 if chosen[0] else 1
            X = S.ray_plane(cams[n], S.undistort(cams[n], [chosen[n][0]])[0], tgt["h"])
            how = f"cam{n} 단독 추정"
        tgt["boxes"] = [c[1] if c else None for c in chosen]
        if X is not None:
            kf.predict(now - tgt["last"]); kf.update(X[:2])
            tgt["last"], tgt["how"] = now, how
            tgt["trail"].append([round(float(X[0]), 2), round(float(X[1]), 2)])
            if lost:
                log(now, "타겟", "다시 찾음 (임베딩 재식별)")
    pos = kf.at(now - tgt["last"])
    state["target"] = {"xyz": [round(float(pos[0]), 2), round(float(pos[1]), 2), round(tgt["h"], 2)],
                       "v": [round(float(kf.x[2]), 2), round(float(kf.x[3]), 2)],
                       "how": tgt["how"] + (" (예측)" if now - tgt["last"] > 0.3 else ""),
                       "lost": now - tgt["last"] > TGT_LOST_S,
                       "boxes": tgt["boxes"], "trail": list(tgt["trail"]), "scores": []}

# ---------------- 메인 처리 ----------------
def draw(n, res):
    vis = res.plot(labels=False, boxes=False)
    t = state["target"]
    box = t["boxes"][n] if t and n < len(t["boxes"]) else None
    if box:
        x1, y1, x2, y2 = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(vis, "TARGET", (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    for tr in tracks:
        if now_global - tr.last > 0.3:
            continue
        u, v = S.project_distorted(cams[n], [*tr.xy, tr.h if args.mode != "ankle1" else 0.0])
        if 0 <= u < W and 0 <= v < H:
            col = (0, 0, 255) if tr.alarm.zone == "near" else (0, 255, 0)
            cv2.circle(vis, (int(u), int(v)), 6, col, -1)
            cv2.putText(vis, f"#{tr.id}", (int(u) + 8, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.putText(vis, f"cam{n} | {args.mode}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()

retarget = threading.Event(); retarget.set()
now_global = 0.0

def process_loop():
    global now_global
    srcs = [Cam(d) for d in args.cams[:NUM_CAMS]]
    for s in srcs:
        s.start()
    t0, n_frames, idx = time.time(), 0, 0
    while True:
        got = [s.get() for s in srcs]
        if any(f is None for _, f in got):
            time.sleep(0.05); continue
        frames = [f for _, f in got]
        if retarget.is_set():
            set_target(frames)
        skew = abs(got[0][0] - got[-1][0])
        results = det.predict(frames, conf=0.5, half=True, verbose=False)
        kps = [np.concatenate([r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()[..., None]], -1)
               if r.keypoints is not None and r.keypoints.conf is not None else np.zeros((0, 17, 3)) for r in results]
        now = now_global = time.time()
        idx += 1
        if args.moving:
            track_target(frames, now, idx)
        obs = []
        if args.mode == "ankle1":
            for k in kps[0]:
                a = S.ankle(k)
                X = S.ray_plane(cams[0], S.undistort(cams[0], [a])[0], 0.0) if a is not None else None
                if X is not None:
                    obs.append((X[:2], 0.0, "ankle"))
        else:
            pts = [[(i, S.upper_body(k)) for i, k in enumerate(kk)] for kk in kps]
            pts = [[(i, p) for i, p in pp if p is not None] for pp in pts]
            und = [S.undistort(cams[n], [p for _, p in pts[n]]) for n in range(2)]
            matched, rest0, rest1 = ([], set(range(len(und[0]))), set(range(len(und[1]))))
            if skew <= MAX_SKEW:
                matched, rest0, rest1 = S.match_people(cams[0], cams[1], und[0], und[1])
            for i, j, X, err in matched:
                obs.append((X[:2], float(X[2]), "stereo"))
            if args.mode == "fused":
                singles = []
                for n, rest in ((0, rest0), (1, rest1)):
                    for i in rest:
                        X = S.ray_plane(cams[n], und[n][i], H_GUESS)
                        if X is not None:
                            singles.append(X[:2])
                for xy in singles:   # 삼각측량 결과나 이미 추가한 추정과 겹치면 같은 사람으로 보고 버림
                    if all(np.linalg.norm(xy - o[0]) > 0.6 for o in obs):
                        obs.append((xy, H_GUESS, "single"))
        x0, y0, x1, y1 = ROOM
        obs = [o for o in obs if x0 <= o[0][0] <= x1 and y0 <= o[0][1] <= y1]
        update_tracks(obs, now)

        people = []
        txy = np.array(state["target"]["xyz"][:2]) if state["target"] else None
        for tr in tracks:
            d = float(np.linalg.norm(tr.xy - txy)) if txy is not None else None
            closing, ttc = 0.0, None
            if d is not None:
                msgs = tr.alarm.update(now, d)
                if args.moving and tgt["kf"] is not None and not state["target"].get("lost"):
                    t_state = np.concatenate([txy, tgt["kf"].x[2:]])
                    p_state = np.concatenate([tr.kf.at(now - tr.last), tr.kf.x[2:]])
                    d, closing, ttc = S.relative_state(p_state, t_state)
                    msgs += tr.alarm.update_ttc(closing, ttc, d)
                for msg in msgs:
                    log(now, f"#{tr.id}", msg)
            people.append({"id": tr.id, "x": round(float(tr.xy[0]), 2), "y": round(float(tr.xy[1]), 2),
                           "h": round(tr.h, 2), "kind": tr.kind,
                           "pose": "-" if tr.kind != "stereo" else ("앉음" if tr.h < SIT_H else "섬"),
                           "d": None if d is None else round(d, 2), "trend": tr.alarm.trend, "zone": tr.alarm.zone,
                           "v": [round(float(tr.kf.x[2]), 2), round(float(tr.kf.x[3]), 2)],
                           "closing": round(closing, 2), "ttc": None if ttc is None else round(ttc, 1),
                           "danger": tr.alarm.ttc_on})
        n_frames += 1
        if now - t0 >= 1:
            state["stats"] = {"fps": round(n_frames / (now - t0), 1), "skew_ms": round(skew * 1000),
                              "mode": args.mode + (" +moving" if args.moving else "")}
            t0, n_frames = now, 0
        state["people"] = people
        for n in range(NUM_CAMS):
            jpegs[n] = draw(n, results[n])

app = FastAPI()

@app.get("/stream/{n}")
def stream(n: int):
    def gen():
        while True:
            f = jpegs[n] if n < len(jpegs) else None
            if f:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
            time.sleep(0.07)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/state")
def get_state():
    with ev_lock:
        ev = list(events)[:25]
    return {**state, "room": ROOM, "near": args.near, "far": args.far, "events": ev, "cams": NUM_CAMS,
            "moving": args.moving,
            "cam_pos": [c["C"].round(2).tolist() for c in cams]}

@app.post("/retarget")
def do_retarget():
    retarget.set(); return {"ok": True}

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>강의실 접근 감지</title>
<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}.row{display:flex;gap:10px;flex-wrap:wrap}
.row>div{flex:1;min-width:300px}img{width:100%;border-radius:6px}svg{background:#1b1b1b;border-radius:6px;width:100%;max-width:640px}
table{border-collapse:collapse}td,th{border:1px solid #444;padding:3px 8px;font-size:13px}button{font-size:14px;padding:4px 10px}
.near{color:#ff5555;font-weight:bold}</style>
<h2>강의실 접근 감지 <span id=st></span></h2>
<p><button onclick="fetch('/retarget',{method:'POST'})">타겟 다시 찾기</button>
<button onclick="snd=true;beep()">알람 소리 켜기</button> <span id=tg></span></p>
<div class=row><div><img src="/stream/0"></div><div id=c1><img src="/stream/1"></div></div>
<div class=row><div><h3>평면도</h3><svg id=map viewBox="0 0 640 480"></svg></div>
<div><h3>사람</h3><table><thead><tr><th>ID</th><th>거리(m)</th><th>추세</th><th>구역</th><th>접근(m/s)</th><th>TTC(s)</th><th>자세</th><th>방식</th></tr></thead><tbody id=pp></tbody></table>
<h3>기록</h3><table><thead><tr><th>시각</th><th>대상</th><th>내용</th></tr></thead><tbody id=ev></tbody></table></div></div>
<script>
let snd=false,lastEv=0;
function beep(){try{const a=new AudioContext(),o=a.createOscillator();o.frequency.value=880;o.connect(a.destination);o.start();o.stop(a.currentTime+0.3);}catch(e){}}
async function poll(){const r=await (await fetch('/state')).json();
 const [x0,y0,x1,y1]=r.room,s=Math.min(600/(x1-x0),440/(y1-y0)),X=x=>20+(x-x0)*s,Y=y=>20+(y-y0)*s;
 const arrow=(x,y,v,c)=>r.moving&&Math.hypot(...v)>0.15?`<line x1=${X(x)} y1=${Y(y)} x2=${X(x+v[0])} y2=${Y(y+v[1])} stroke=${c} stroke-width=2 />`:'';
 if(r.cams<2)document.getElementById('c1').style.display='none';
 document.getElementById('st').textContent=r.stats.fps?`[${r.stats.mode}] ${r.stats.fps} fps, 카메라 시간 차 ${r.stats.skew_ms} ms`:'';
 let g=`<rect x=${X(x0)} y=${Y(y0)} width=${(x1-x0)*s} height=${(y1-y0)*s} fill=none stroke=#555 />`;
 r.cam_pos.forEach((c,i)=>g+=`<rect x=${X(c[0])-6} y=${Y(c[1])-6} width=12 height=12 fill=#4af /><text x=${X(c[0])+8} y=${Y(c[1])} fill=#4af font-size=12>cam${i}</text>`);
 const t=r.target;
 if(t){const tx=X(t.xyz[0]),ty=Y(t.xyz[1]);
  g+=`<circle cx=${tx} cy=${ty} r=${r.far*s} fill=none stroke=#666 stroke-dasharray=4 /><circle cx=${tx} cy=${ty} r=${r.near*s} fill=#ff000022 stroke=#f55 />`+
     `<rect x=${tx-7} y=${ty-7} width=14 height=14 fill=#f0f /><text x=${tx+10} y=${ty-8} fill=#f0f font-size=12>TARGET</text>`;
  if(t.trail&&t.trail.length>1)g+=`<polyline points="${t.trail.map(p=>X(p[0])+','+Y(p[1])).join(' ')}" fill=none stroke=#f0f stroke-opacity=0.5 />`;
  if(t.v)g+=arrow(t.xyz[0],t.xyz[1],t.v,'#f0f');
  document.getElementById('tg').textContent=`타겟 ${t.xyz.join(', ')} (${t.how})`+(t.v?` 속도 ${Math.hypot(...t.v).toFixed(2)}m/s`:'')+(t.lost?' — 놓침':'');}
 else document.getElementById('tg').textContent='타겟 없음';
 r.people.forEach(p=>{const c=(p.zone=='near'||p.danger)?'#f55':p.kind=='single'?'#aa0':'#5f5';
  g+=arrow(p.x,p.y,p.v,c);
  g+=`<circle cx=${X(p.x)} cy=${Y(p.y)} r=8 fill=${c} ${p.kind=='single'?'fill-opacity=0.5':''} /><text x=${X(p.x)+10} y=${Y(p.y)+4} fill=${c} font-size=13>#${p.id} ${p.d??''}</text>`;});
 document.getElementById('map').innerHTML=g;
 document.getElementById('pp').innerHTML=r.people.map(p=>`<tr class=${(p.zone=='near'||p.danger)?'near':''}><td>#${p.id}</td><td>${p.d??'-'}</td><td>${p.trend}</td><td>${p.zone}</td><td>${p.closing}</td><td>${p.ttc??'-'}</td><td>${p.pose}</td><td>${p.kind}</td></tr>`).join('');
 document.getElementById('ev').innerHTML=r.events.map(e=>`<tr><td>${new Date(e.ts*1000).toLocaleTimeString()}</td><td>${e.who}</td><td>${e.label}</td></tr>`).join('');
 const fresh=r.events.filter(e=>e.ts>lastEv&&e.label.startsWith('알람'));if(fresh.length&&snd)beep();
 if(r.events.length)lastEv=Math.max(lastEv,r.events[0].ts);}
setInterval(poll,500);poll();
</script>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE

if __name__ == "__main__":
    threading.Thread(target=process_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
```

### 10-3 카메라 1대 + 발목 (문제 확인)

타겟 위치는 먼저 줄자로 잰 값을 `--target-xy`로 주고, 위치 계산만 비교한다. 바닥에 타겟에서 0.5m·1m·2m·3m 떨어진 지점을 테이프로 표시해 둔다.

1. `ankle1` 모드로 실행하고 `http://<Jetson IP>:8000`에 접속한다.
2. 테이프 지점마다 서서, 화면 거리값을 10-6 표의 ankle1 열에 적는다.
3. 같은 지점 근처 책상에 앉거나 책상 뒤에 서서 다시 잰다.

```bash
cd ~/eg2-lab/prox && source ~/eg2/bin/activate
python proximity_app.py --cams 0 2 --mode ankle1 --target-xy 3.0,1.0
```

확인할 문제:

- 서 있을 때는 거리가 대체로 맞지만, 앉거나 책상 뒤에 서면 발목이 가려져 사람이 사라지거나, 책상 다리·가방을 발목으로 잡아 위치가 크게 튀다.
- 카메라에서 멀수록 발목 1\~2px 오차가 수십 cm로 커진다.
- 사람이 다른 사람에 가려지면 추적이 끊기고 ID가 바뀐다.

### 10-4 카메라 2대 + 상체 삼각측량 (개선)

두 카메라의 어깨 중심을 삼각측량해 바닥 위치와 높이를 함께 구한다. 높이가 1.15m 미만이면 "앉음", 이상이면 "섬"으로 표시된다.

1. `stereo` 모드로 실행하고 10-3과 같은 지점·자세로 다시 잰다.
2. 화면 상단의 "카메라 시간 차"가 80ms 이하인지 본다. 넘으면 그 순간은 삼각측량을 건너뛴다.
3. 한 카메라에만 보이는 위치(교실 가장자리, 다른 사람 뒤)로 가서 사람이 사라지는지 확인한다.

```bash
python proximity_app.py --cams 0 2 --mode stereo --target-xy 3.0,1.0
```

확인할 점: 앉아도 위치가 유지되고 거리 오차가 줄어든다. 대신 두 카메라 모두에 보여야 하므로, 한쪽에서 가려지면 사람이 사라지고 ID가 끊긴다.

### 10-5 한 카메라 추정으로 보완 (fused)

두 카메라에서 짝을 찾지 못한 사람은 어깨가 높이 1.2m에 있다고 보고 위치를 추정한다. 평면도에는 반투명 노란 점, 표에는 `single`로 표시된다.

1. 기본(`fused`) 모드로 실행하고, 10-4에서 사람이 사라지던 위치를 다시 걸어 본다.
2. `single`로 표시된 순간의 거리 오차를 기록한다.

```bash
python proximity_app.py --cams 0 2 --target-xy 3.0,1.0
```

확인할 점: 추적이 끊기지 않는 대신, 추정값은 오차가 크다. 가상 배치 시뮬레이션에서 삼각측량은 5cm 안팎이었지만, 한 카메라 추정은 사람의 실제 어깨 높이가 1.2m에서 0.2m만 달라도 0.6\~1.2m 빗나갔다. 그래서 추정값은 알람보다 "추적 유지"용으로 보고, 알람을 추정값에서 내지 않게 바꾸는 것이 확장 과제다.

### 10-6 타겟 자동 찾기와 접근 알람

이제 `--target`으로 타겟을 문구로 지정한다. YOLOE가 후보 박스를 내고, EmbeddingGemma 2가 후보 크롭과 문구의 유사도로 하나를 고른 뒤, 두 카메라 박스 중심을 삼각측량한다. 두 모델은 타겟을 찾은 뒤 메모리에서 내린다. 자리를 옮기면 웹의 **타겟 다시 찾기**를 누른다.

1. 실행 후 기록의 "타겟 찾음" 줄에서 좌표와 방식(삼각측량/단독 추정), 점수를 확인하고 줄자 실측값과 비교한다.
2. **알람 소리 켜기**를 누른다 (브라우저는 클릭이 있어야 소리를 낼 수 있다).
3. 아래 시나리오를 각 3회씩 수행하고 결과를 적는다.

```bash
python proximity_app.py --cams 0 2 --target "red fire extinguisher" --near 1.0 --far 3.0
```

| 시나리오 (3회씩) | 기대 결과 | 맞은 횟수 | 비고 |
| --- | --- | --- | --- |
| 4m에서 타겟으로 걸어옴 | 접근 시작 → 알람: 근접 |  |  |
| 타겟 앞 0.5m에서 4m 밖으로 나감 | 알람: 이탈 |  |  |
| 1.0m 경계에서 10초 서 있음 | 알람 1회만 (깜빡임 없음) |  |  |
| 타겟 2m 옆을 지나감 | 근접 알람 없음 |  |  |
| 타겟 옆 책상에 앉음 | 근접 알람, 자세 = 앉음 |  |  |

### 10-7 결과 비교

같은 테이프 지점에서 타겟까지의 화면 거리(m)를 적는다. 사라지면 "없음"으로 적는다.

| 실제 거리·자세 | ankle1 (10-3) | stereo (10-4) | fused (10-5) |
| --- | --- | --- | --- |
| 0.5m, 서 있음 |  |  |  |
| 1.0m, 서 있음 |  |  |  |
| 2.0m, 서 있음 |  |  |  |
| 3.0m, 서 있음 |  |  |  |
| 1.0m, 책상에 앉음 |  |  |  |
| 2.0m, 책상 뒤에 서 있음 |  |  |  |
| 한 카메라에만 보이는 위치 |  |  |  |

| 모드 | fps | 카메라 시간 차 (ms) | RAM (MB) |
| --- | --- | --- | --- |
| ankle1 |  |  |  |
| stereo |  |  |  |
| fused |  |  |  |

확인 포인트:

- 알람 기준은 `--near`·`--far`, 히스테리시스 폭은 `ProximityAlarm(margin=0.2)`, 접근 판정 속도는 `v_th=0.3`(m/s)으로 조정한다.
- 사람 두 명이 가까이 있으면 잘못 매칭될 수 있다. 재투영 오차 기준 `max_err=15`(px)을 낮추면 오매칭은 줄고 누락은 늘어난다.
- 카메라가 조금이라도 움직이면 검증점 오차가 커진다. 수업 전 10-2 화면에서 계산만 다시 눌러 오차를 확인하는 습관을 들인다.
- 사람 이미지는 저장하지 않는다. 남는 것은 바닥 좌표와 알람 기록뿐이다.
- 확장 과제: 알람을 MQTT로 발행해 환경 모니터링 대시보드에 연동하거나, 타겟을 여러 개로 늘려 본다.

### 10-8 이동 타겟 (`--moving`)

타겟이 움직이는 경우(수업용 카트, 로봇, 다른 사람)에는 타겟도 매 프레임 추적해 사람과의 거리와 상대 속도를 구한다. 거리만 보는 근접 알람에 더해, "이 속도라면 몇 초 뒤 부딪히는가"(충돌 예상 시간, TTC)로 더 일찍 알린다.

| 단계 | 처리 | 코드 |
| --- | --- | --- |
| 1. 검출 | 2프레임마다(`--target-every`) 두 카메라에서 YOLOE로 타겟 후보 박스 | `track_target()` |
| 2. 같은 타겟 고르기 | 칼만 필터 예측 위치를 각 카메라 화면에 투영해 100px 이내 가장 가까운 후보. 2초 이상 놓치면 처음 찾은 타겟의 EmbeddingGemma 2 임베딩과 가장 닮은 후보로 재식별 | `tgt["ref"]` |
| 3. 위치 | 두 카메라 박스 중심 삼각측량 (한쪽만 보이면 마지막 타겟 높이 평면으로 추정) | `S.triangulate()` |
| 4. 위치·속도 추정 | 타겟과 사람 모두 칼만 필터 (x, y, vx, vy). 프레임을 놓쳐도 예측값으로 이어 감 | `S.Kalman2D` |
| 5. 알람 | 거리 알람(근접·이탈) + 접근 속도 > 0.5m/s이고 TTC < 2초(`--ttc`)면 "충돌 위험" | `relative_state()`, `ProximityAlarm.update_ttc()` |

```latex
\text{TTC} = \frac{d}{v_{\text{접근}}},\qquad v_{\text{접근}} = -\frac{(\mathbf{p}_t-\mathbf{p}_h)\cdot(\mathbf{v}_t-\mathbf{v}_h)}{\lVert \mathbf{p}_t-\mathbf{p}_h \rVert}
```

여기서 p는 바닥 위치, v는 속도, t는 타겟, h는 사람이다. 둘 중 누가 움직여도 같은 식으로 계산되고, 고정 타겟이면 타겟 속도가 0일 뿐이다.

웹 화면 변화: 평면도에 타겟 이동 경로(보라 선)와 타겟·사람의 속도 화살표(1초 뒤 위치)가 그려지고, 사람 표에 접근 속도와 TTC가 추가된다. 충돌 위험인 사람은 빨간색으로 표시된다. YOLOE와 EmbeddingGemma를 계속 올려 두므로 고정 타겟 모드보다 메모리를 더 쓰고 fps가 낮아진다.

**속도에 따른 한계**: 두 카메라 시간 차 80ms 기준 위치 오차는 대략 속도 × 0.08초다. 걸음 속도(1m/s)는 약 8cm로 문제없고, 3m/s는 약 24cm로 경계, 공·자전거처럼 10m/s 이상은 웹캠으로는 부적합하다. 강의실에서는 사람 걸음 속도의 대상까지가 현실적인 범위다.

1. 움직일 타겟(바퀴 달린 카트, 상자를 올린 의자 등)을 정하고 문구를 지정해 실행한다. 처음에는 타겟을 멈춘 상태에서 "타겟 찾음"을 확인한다.
2. 사람은 제자리에 서 있고, 다른 사람이 카트를 4m에서 천천히(약 0.5m/s), 보통(약 1m/s), 빠르게(약 1.5m/s) 밀어 다가간다. 카트를 미는 사람은 타겟과 거의 같은 자리에 있으므로 알람 표에서 제외하고 본다.
3. "충돌 위험" 알람과 "근접" 알람의 시각 차이, 알람 때의 거리를 기록한다.
4. 카트를 한쪽 카메라에서 가려지게 지나가게 해 "예측", "놓침", "다시 찾음 (임베딩 재식별)"이 어떻게 나오는지 본다.

```bash
python proximity_app.py --cams 0 2 --target "cart" --moving
python proximity_app.py --cams 0 2 --target "cart" --moving --ttc 3.0     # 더 일찍 알람
```

| 카트 속도 | 충돌 위험 알람 때 거리 (m) | 근접 알람보다 먼저 울린 시간 (초) | 추정 속도 (m/s) | 비고 |
| --- | --- | --- | --- | --- |
| 천천히 (약 0.5m/s) |  |  |  | 접근 속도 기준(0.5) 근처라 알람이 안 울릴 수 있음 |
| 보통 (약 1m/s) |  |  |  |  |
| 빠르게 (약 1.5m/s) |  |  |  |  |
| 한 카메라에서 가려짐 |  |  |  | 예측·재식별 동작 |

확인 포인트:

- 가상 시뮬레이션(카트 1.2m/s, 약 7fps, 위치 잡음 5cm)에서 충돌 위험 알람은 2.2m 지점에서, 근접 알람보다 약 1초 먼저 울렸다. 칼만 필터가 속도를 따라잡는 데 0.5초 정도 걸렸다.
- 같은 시뮬레이션에서 카트가 갑자기 멈추자, 필터가 멈춤을 늦게 반영해 거리가 잠시 실제보다 짧게(최대 약 20cm) 나왔다. `Kalman2D(q=...)`를 키우면 반응이 빨라지고 대신 위치가 더 흔들린다. 이 균형을 실험해 보는 것이 과제다.
- 비슷하게 생긴 물체가 여럿이면 추적이 다른 물체로 넘어갈 수 있다. 타겟에 색 테이프를 붙이고 문구에 색을 넣으면(예: "cart with yellow tape") 재식별이 안정된다.
- `--moving`에는 `--target` 문구가 필요하다. 매 프레임 YOLOE로 타겟을 찾기 때문이다.

## Q&A: 설치 중 실제로 겪은 오류

실습 장비(xtra-jetson01)에서 나온 오류를 나온 순서대로 정리했다. 대부분은 JetPack 버전과 설치 절차가 맞지 않아서 생긴다.

**Q1. `pip install torch --index-url .../jp6/cu126`에서 `Could not find a version that satisfies the requirement torch (from versions: none)`가 나온다.**

A. 인덱스에 있는 휠이 현재 Python 버전과 맞지 않는 경우다. jp6/cu126 휠은 전부 Python 3.10(`cp310`)용이라, Python 3.12에서는 설치할 후보가 하나도 없다. `python -V`로 버전을 확인하고, 3.12이면 JetPack 7 경로(cu132 인덱스)를 쓴다.

**Q2. `curl -sI` 결과가 `200`이 아니라 `302 Moved Temporarily`다. 네트워크 문제인가?**

A. 아니다. 인덱스 서버가 다른 주소로 넘겨주는 정상 응답이고, pip는 이를 자동으로 따라간다. `curl -sIL <주소> | grep HTTP`로 마지막 응답이 200인지 보면 된다.

**Q3. `/usr/bin/python3.10: No such file or directory`**

A. Ubuntu 24.04(JetPack 7)에는 Python 3.10이 기본으로 설치되어 있지 않다. 3.10을 따로 설치하지 말고, 시스템 Python 3.12로 venv를 만든 뒤 JetPack 7용 torch를 설치한다.

**Q4. `error: externally-managed-environment`**

A. venv가 활성화되지 않은 상태에서 시스템 pip가 실행된 것이다. Q3처럼 venv 생성이 실패하면 바로 이어지는 오류다. `--break-system-packages`는 쓰지 말고 venv를 다시 만든다.

```bash
sudo apt install -y python3-venv python3-full
rm -rf ~/eg2 && python3 -m venv ~/eg2 && source ~/eg2/bin/activate
which pip        # ~/eg2/bin/pip 이어야 함
```

**Q5. `nvcc: command not found`**

A. CUDA 툴킷은 설치되어 있지만 PATH에 등록되지 않은 경우가 많다. `ls -d /usr/local/cuda*`로 폴더가 있는지 확인한다.

```bash
# 폴더가 있으면 PATH 등록
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc && nvcc --version

# 폴더가 없으면 JetPack 구성요소 설치 (수 GB)
sudo apt update && sudo apt install -y nvidia-jetpack
```

nvcc가 없어도 torch는 동작할 수 있다. 다만 JetPack 버전 판단은 `cat /etc/nv_tegra_release`로 한다.

**Q6. 내 장비가 JetPack 6인지 7인지 빠르게 구분하려면?**

A. 아래 세 줄의 결과로 판단하고, 1단계 5번 표에서 맞는 인덱스를 고른다.

```bash
cat /etc/nv_tegra_release      # R36 = JetPack 6, R38 = JetPack 7
lsb_release -rs                # 22.04 = JetPack 6, 24.04 = JetPack 7
python3 -V                     # 3.10 = JetPack 6, 3.12 = JetPack 7
```

**Q7. `check.py`에서 `PyTorch has CUDA version 13.2 whereas TorchAudio has CUDA version 13.0`, 이어서 `Could not import module 'AutoProcessor'`**

A. torch는 cu132 인덱스에서, torchaudio는 PyPI(cu130)에서 설치돼 CUDA 버전이 어긋난 것이다. cu132 인덱스에는 Python 3.12용 최신 torchaudio가 없다. transformers는 torchaudio가 설치돼 있으면 import하므로, 이 오류가 sentence-transformers 전체를 막는다.

방법 1 (권장): torchaudio를 제거한다. 오디오는 soundfile·librosa로 읽는다.

```bash
pip uninstall -y torchaudio
pip install -c ~/eg2/torch-pin.txt soundfile librosa
python check.py
```

방법 2: 5단계 음성 테스트에서 torchaudio가 꼭 필요하다고 나오면, 세 패키지를 모두 PyPI의 cu130 빌드로 맞춘다. 반드시 커널 실행 검증(1단계 5번)을 다시 한다.

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio      # PyPI aarch64 = CUDA 13.0 빌드
pip freeze | grep -E '^(torch|torchvision|torchaudio)==' > ~/eg2/torch-pin.txt
```

**Q8. 모델 로드 시 `Warning: You are sending unauthenticated requests to the HF Hub` 후 한참 멈춘 것처럼 보인다.**

A. 오류가 아니라 경고다. 토큰 없이도 모델을 받을 수 있고, 멈춘 것처럼 보이는 동안 모델 파일(수 GB)을 내려받는 중이다. 다른 터미널에서 캐시 크기가 늘어나는지 확인한다.

```bash
watch -n 5 du -sh ${HF_HOME:-~/.cache/huggingface}/hub
```

속도 제한을 피하려면 [Hugging Face 토큰](https://huggingface.co/settings/tokens)(Read 권한)으로 로그인한다. 다운로드가 끝난 뒤에는 오프라인 모드로 실행하면 네트워크 확인 없이 바로 로드된다.

```bash
hf auth login                     # 토큰 붙여넣기 (선택)
hf download google/embeddinggemma-2   # 진행률 표시하며 미리 받기
echo 'export HF_HUB_OFFLINE=1' >> ~/.bashrc   # 다운로드 완료 후
```

**Q9. `train_ft.py`에서 `BaseModel.gradient_checkpointing_enable() got an unexpected keyword argument 'every_n_layers'`**

A. Transformers v5 Trainer가 넘기는 인자를 sentence-transformers 모델 래퍼가 받지 못해서 생기는 버전 불일치다. `SentenceTransformerTrainingArguments`에서 `gradient_checkpointing=True`를 빼고, 내부 HF 모델에 직접 켠다 (7단계 학습 스크립트 반영). 같은 실행에서 나온 import 경로 `DeprecationWarning`과 `warmup_ratio` 경고도 함께 고쳤다.

```python
hf = getattr(model[0], "auto_model", None) or getattr(model[0], "model", None)
hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
```

**Q10. 학습 로그가 `train_runtime` 출력과 `Writing model shards 100%` 뒤에서 멈춰 있다.**

A. 학습과 저장이 끝난 상태다. `tail -f`는 로그가 더 쌓이길 계속 기다리는 명령이라 스스로 끝나지 않는다. Ctrl+C로 빠져나온 뒤 프로세스 종료와 모델 파일을 확인한다.

```bash
jobs                         # [1]+ Done ... 이면 정상 종료
ls -lh models/eg2-classroom  # model.safetensors, config 등이 있어야 함
python eval_ir.py models/eg2-classroom
```

**Q11. 8단계 `app.py`에서 `웹캠을 열 수 없음` 또는 웹 화면이 검게 나온다.**

A. 장치 번호, 권한, 다른 프로세스 점유 순서로 확인한다. 웹캠이 `/dev/video1`이면 `CAM = 1`로 바꾼다. MJPG를 지원하지 않는 웹캠은 `CAP_PROP_FOURCC` 줄을 지우고 640x480으로 설정한다.

```bash
v4l2-ctl --list-devices
groups | grep video || sudo usermod -aG video $USER   # 추가 후 다시 로그인
sudo fuser /dev/video0                                 # 카메라를 잡고 있는 프로세스
```

웹 페이지 자체가 열리지 않으면 Jetson IP와 방화벽(`sudo ufw allow 8000/tcp`)을 확인한다.

**Q12. 8단계 설치 중 `pyctcdecode 0.5.0 requires numpy<2.0.0 ... but you have numpy 2.x`**

A. 설치는 끝난 상태이고 경고만 남은 것이다. pyctcdecode는 음성 인식(CTC) 디코딩용 부가 패키지로, 이 실습의 어떤 코드도 쓰지 않는다. numpy를 1.x로 내리면 OpenCV·ultralytics가 깨질 수 있으므로, numpy는 그대로 두고 pyctcdecode를 제거한다.

```bash
pip uninstall -y pyctcdecode
pip check                      # 남은 충돌 확인
python -c "import cv2, numpy, ultralytics; print(cv2.__version__, numpy.__version__)"
```

**Q13. 드럼 세탁기를 `oven`, 원통형 전지 포장을 `cell phone`으로 인식한다.**

A. YOLO 기본 모델(COCO 80종)에 세탁기와 전지 클래스가 없어서 생기는 현상이다. 단계별로 대응한다.

| 방법 | 난이도 | 효과 |
| --- | --- | --- |
| 1. 임베딩 재라벨 (8단계 코드에 반영) | 낮음 | 박스가 잡히면 라벨만 바로잡음. 추가 학습 없음 |
| 2. 오픈 어휘 검출기 (YOLOE / YOLO-World) | 중간 | 텍스트로 클래스 지정 → 박스까지 새 클래스로 검출 |
| 3. YOLO 커스텀 학습 | 높음 | 클래스당 수백 장 라벨링 필요, 정확도 가장 높음 |

방법 2는 모델 로드 부분만 바꾼다 (ultralytics의 YOLOE 기준, 모델 파일명과 API는 설치된 버전 문서로 확인).

```python
from ultralytics import YOLOE
names = ["drum washing machine", "battery pack", "cell phone", "oven", "laptop", "cup"]
det = YOLOE("yoloe-11s-seg.pt")
det.set_classes(names, det.get_text_pe(names))
```

**Q14. 10단계에서 카메라 2대 중 하나가 열리지 않거나, 둘째 카메라 영상이 멈춘다.**

A. 대부분 장치 번호나 USB 대역폭 문제다. 웹캠 하나가 `/dev/video` 두 개(영상용과 메타데이터용)를 만드는 경우가 많아, 카메라 2대면 보통 0과 2가 실제 영상 장치다. 두 카메라가 같은 USB 컨트롤러를 쓰면 비압축(YUYV) 형식에서는 대역폭이 모자라 둘째가 열리지 않는다.

```bash
v4l2-ctl --list-devices                               # 카메라별 찫 번째 /dev/videoN이 영상 장치
v4l2-ctl -d /dev/video2 --list-formats-ext | grep -A2 MJPG   # MJPG 640x480 지원 확인
lsusb -t                                              # 두 카메라가 같은 버스(Bus)에 있는지 확인
```

코드는 MJPG 640x480으로 설정하므로 보통 두 대가 함께 열린다. 그래도 안 되면 두 카메라를 서로 다른 USB 포트(가능하면 다른 버스)에 꼽거나, 전원이 공급되는 USB 허브를 쓴다. 5m가 넘는 연장선은 리피터가 들어간 능동형을 쓴다.

## 참고 자료

- [google/embeddinggemma-2 · Hugging Face](https://huggingface.co/google/embeddinggemma-2)
- [EmbeddingGemma 2: The Developer Guide — Google Developers Blog](https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/)
- [ggml-org/embeddinggemma-2-GGUF · Hugging Face](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF)
- [jetson-containers](https://github.com/dusty-nv/jetson-containers)