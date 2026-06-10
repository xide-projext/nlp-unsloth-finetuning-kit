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

## 원논문 (선택, 리포트 인용용)

- **LoRA**: Hu et al., 2021 — *"LoRA: Low-Rank Adaptation of Large Language Models"* (arXiv:2106.09685)
- **QLoRA**: Dettmers et al., 2023 — *"QLoRA: Efficient Finetuning of Quantized LLMs"* (arXiv:2305.14314, NF4 출처)

---

## 이 키트 안에서

- 개념 직관 → [`01-core-concepts.md`](01-core-concepts.md) + `tutorial/*.py`
- 손잡이 → [`02-hyperparameters.md`](02-hyperparameters.md)
- 데이터 → [`03-dataset-design.md`](03-dataset-design.md)
- 실행 → [`04-unsloth-studio-workflow.md`](04-unsloth-studio-workflow.md)
