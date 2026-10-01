# tiny_llm

GPU 한 장(RTX 3060 12GB)으로 작은 언어모델을 처음부터 직접 만들어 본 프로젝트입니다.
쓸 만한 모델을 얻는 것보다 **LLM 의 구조와 동작을 직접 확인하는 것**이 목적입니다.

- 데이터: [TinyStoriesV2-GPT4](https://huggingface.co/datasets/roneneldan/TinyStories)(어린이 동화 271만 편, 약 5.5억 토큰)
- 토크나이저: 직접 학습한 SentencePiece BPE, vocab 4000
- 구조: Llama 형(RMSNorm / RoPE / SwiGLU), 문맥 길이 512

## 학습된 모델 3개

| | 파라미터 | 구조 | 학습 토큰 | 3060 학습 시간 | val loss |
|---|---|---|---|---|---|
| A | 1.3M | d=128, 4층, head 4 | 26M | 1.2분 | 2.042 |
| B | 7.5M | d=256, 8층, head 8 | 147M | 23.5분 | 1.387 |
| C | 27.0M | d=512, 8층, head 8 | 545M (1에폭) | 176.6분 | 1.138 |

가중치는 `runs/*/` 아래에 들어 있습니다. C 는 GitHub 의 100MB 파일 한도 때문에 fp16 으로 바꾼 `ckpt_fp16.pt` 만 올렸습니다.

## 바로 해보기 (데이터 없이)

```bash
pip install torch sentencepiece          # GPU 가 없으면 CPU 로 돌아갑니다
python gen.py -r C "Once upon a time"    # C 모델로 이어 쓰기
python gen.py -r B                       # 프롬프트를 빼면 대화형
python compare.py "The dragon was very hungry."   # A/B/C 를 같은 시드로 나란히 비교
```

- C 모델은 `-r C` 로 지정해야 합니다. 생략하면 가장 최근의 `ckpt.pt`(B)를 고르고,
  `gen.py --list` 에도 fp16 인 C 는 나오지 않습니다.
- Windows 콘솔에서 글자가 깨지거나 오류가 나면 `PYTHONIOENCODING=utf-8` 을 붙여 실행하세요.

## 처음부터 학습하기

```bash
cd data && bash dl.sh && cd ..   # TinyStories 원문 다운로드 (약 2.2GB)
python pack.py                   # 텍스트 → 토큰 배열 data/train.bin, data/val.bin
python train.py --preset A       # A/B/C 중 선택. 결과는 runs/<시각>_<preset>/
```

CUDA 용 torch 설치는 `install_torch.sh` 를 참고하세요(cu126 기준).

## 파일 안내

| 파일 | 용도 |
|---|---|
| `model.py` | 모델 정의 |
| `data.py` | 학습 배치 생성 |
| `train.py` | 학습(`--preset A/B/C`, `--no_ffn`) |
| `pack.py` | 텍스트 → 토큰 `.bin` |
| `gen.py` / `compare.py` | 생성 / 여러 모델 비교 |
| `trace.py` | 문장 하나가 모델을 통과할 때 텐서 모양 출력 |
| `tok_compare.py` | 토크나이저 vocab 4k/8k/16k 비교 |
| `epoch_exp.py` | 같은 데이터를 여러 번 볼 때의 과적합 실험 |
| `inspect_attn.py`, `zorbax*.py` | 어텐션 행렬 들여다보기(induction head 탐색) |
| `report.py`, `make_page.py` | 비교 결과를 `REPORT.md` / `artifact.html` 로 |

## 더 읽을 것

- `STATUS.md` — 만든 것, 실험 결과와 결론, 다음에 할 수 있는 것
- `CURRICULUM.md` — 단계별 전체 계획과 용어 정리
- `REPORT.md`, `artifact.html` — A/B/C 에 같은 문장 5개를 넣어 비교한 결과

## 라이선스

- 코드: MIT (`LICENSE`)
- 학습 데이터 TinyStories 는 [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/) 입니다.
  이 데이터로 학습한 가중치(`runs/*/*.pt`)와 생성 샘플을 다시 배포할 때는 그 조건도 함께 확인하세요.
