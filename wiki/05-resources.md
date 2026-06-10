# 05. 추천 영상 · 문서 · 링크

"개념 → 수학적 직관(비유) → 작은 예시" 스타일로 잘 설명한 자료 위주. 비유·시각자료가 강한 것을
골랐습니다(영어가 많지만 그림이 많아 슬라이드 소스로 좋음).

---

## ⭐ 가장 중요 (필독 문서)

리포트의 "왜 이 값을 골랐는가"에 직결됩니다.

| 자료 | 왜 보나 |
|---|---|
| **Unsloth — Fine-tuning LLMs Guide** (공식) | 개념·데이터셋·방법론 전반. 시작점. |
| **Unsloth — LoRA Hyperparameters Guide** (공식) | rank·alpha·epoch·lr 권장값과 *그 이유*를 수백 편 실험 기반으로 정리. 리포트 근거의 핵심. |
| **Unsloth Studio 문서** | <http://unsloth.ai/docs/new/studio> — 과제의 베이스. |
| **DataCamp — Unsloth Studio Fine-Tuning 튜토리얼** | Studio에서 QLoRA 학습 → 채팅 테스트 → GGUF export까지 실제 화면 흐름. 과제 워크플로우와 거의 1:1. |
| **베이스 모델별 가이드** (예: Qwen Fine-tune) | 고른 모델에 맞는 세부 설정. |

---

## 🎬 추천 영상 (기말 수준이면 충분)

| 영상 | 용도 |
|---|---|
| **How to Fine-tune LLMs with Unsloth: Complete Guide** | 전체 워크플로우 종합. 첫 영상으로 추천. |
| **Unsloth Fine-Tuning: LoRA and QLoRA Guide** | LoRA/QLoRA 개념 step-by-step. 개념 보강. |
| **Fine-Tuning Local LLMs with Unsloth & Ollama** | 학습 모델을 Ollama로 로컬 실행/배포. export 단계 이해. |

> YouTube에서 위 제목으로 검색. Unsloth 공식 채널/문서에 최신 링크가 모여 있습니다.

---

## 🧠 개념별 심화 (비유가 좋은 글)

**LoRA · rank 직관**
- Sebastian Raschka, *"Finetuning LLMs with LoRA"* — PCA/SVD 비유로 푸는 사실상의 표준 글.
- ML6, *"LoRA: a technical deep dive"* — 행렬 rank와 분해를 선형대수 기초부터. 강의 흐름과 잘 맞음.

**양자화 (Quantization)**
- Maarten Grootendorst, *"A Visual Guide to Quantization"* — 50+ 다이어그램으로 absmax·NF4까지.
  양자화 자료의 결정판.
- Baseten, *"Four bits"* — JPEG·붓질 비유로 직관 잡기.

**전체 워크플로우 + 하이퍼파라미터**
- Unsloth 공식 Fine-tuning Guide + LoRA Hyperparameters Guide (위 ⭐ 참조).

---

## 📄 대표 논문 마스터 색인 (리포트 인용용)

각 논문의 **설명 방식·직접 예시**는 링크된 위키 섹션에 자세히 적어 두었습니다. 아래는 빠른 색인.

| # | 논문 (연도) | 한 줄 핵심 + 대표 수치 | 자세히 | arXiv |
|---|---|---|---|---|
| 1 | **LoRA** (Hu, 2021) | ΔW를 저랭크 B·A로. GPT-3 175B 학습파라미터 **10,000배↓**, 메모리 3배↓ | [01](01-core-concepts.md)·[02](02-hyperparameters.md) | [2106.09685](https://arxiv.org/abs/2106.09685) |
| 2 | **Intrinsic Dimensionality** (Aghajanyan, 2020) | LoRA의 토대. RoBERTa **200개 파라미터**로 MRPC 90% | [01](01-core-concepts.md) | [2012.13255](https://arxiv.org/abs/2012.13255) |
| 3 | **QLoRA** (Dettmers, 2023) | NF4(정보이론 최적)+이중양자화. **65B를 48GB 1장**에, Guanaco 99.3% | [01](01-core-concepts.md) | [2305.14314](https://arxiv.org/abs/2305.14314) |
| 4 | **LLM.int8()** (Dettmers, 2022) | 이상치 특징 → 혼합정밀도. **6.7B**부터 outlier 창발 | [01](01-core-concepts.md) | [2208.07339](https://arxiv.org/abs/2208.07339) |
| 5 | **GPTQ** (Frantar, 2022) | 학습 없는 PTQ. **175B를 4 GPU시간**에 3~4bit | [01](01-core-concepts.md) | [2210.17323](https://arxiv.org/abs/2210.17323) |
| 6 | **rsLoRA** (Kalajdzievski, 2023) | `α/r`은 rank↑시 gradient 붕괴 → **`α/√r`** | [02](02-hyperparameters.md) | [2312.03732](https://arxiv.org/abs/2312.03732) |
| 7 | **DoRA** (Liu, 2024) | 가중치를 크기·방향 분해, 방향만 LoRA | [02](02-hyperparameters.md) | [2402.09353](https://arxiv.org/abs/2402.09353) |
| 8 | **LIMA** (Zhou, 2023) | Superficial Alignment. **1,000개**로 GPT-4 대비 43% 선호 | [03](03-dataset-design.md) | [2305.11206](https://arxiv.org/abs/2305.11206) |
| 9 | **InstructGPT** (Ouyang, 2022) | RLHF. **1.3B가 175B GPT-3보다 선호** | [03](03-dataset-design.md) | [2203.02155](https://arxiv.org/abs/2203.02155) |
| 10 | **Self-Instruct** (Wang, 2022) | 데이터 부트스트랩. **175 seed → 52K** | [03](03-dataset-design.md) | [2212.10560](https://arxiv.org/abs/2212.10560) |
| 11 | **BPE/Subword** (Sennrich, 2016) | open-vocab 서브워드. 'low'+'er' 병합 | [03](03-dataset-design.md) | [1508.07909](https://arxiv.org/abs/1508.07909) |
| 12 | **SentencePiece** (Kudo, 2018) | 언어독립 분절(`▁`). 한국어 유용 | [03](03-dataset-design.md) | [1808.06226](https://arxiv.org/abs/1808.06226) |

> 인용 팁: 리포트에서 "LoRA를 썼다"보다 "Hu et al.(2021)이 보인 *저랭크 가설*에 근거해 r=16으로
> 설정했다"처럼 **논문→선택의 연결**을 쓰면 점수에 직결됩니다.

---

---

## 이 키트 안에서

- 개념 직관 → [`01-core-concepts.md`](01-core-concepts.md) + `tutorial/*.py`
- 손잡이 → [`02-hyperparameters.md`](02-hyperparameters.md)
- 데이터 → [`03-dataset-design.md`](03-dataset-design.md)
- 실행 → [`04-unsloth-studio-workflow.md`](04-unsloth-studio-workflow.md)
