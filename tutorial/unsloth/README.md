# 실습 (핵심) — 실제 Unsloth로 파인튜닝하기

> 이 폴더가 **본 실습**입니다. 과제의 핵심은 *직접 Unsloth를 돌려 보는 것*이므로,
> 개념을 numpy로 흉내 내는 데서 그치지 않고 **실제 모델을 파인튜닝**합니다.
> (numpy 개념 데모는 [`../concepts/`](../concepts/)로 옮겼고, 어디까지나 *선택적 보충*입니다.)

같은 폴더의 [`sample_data.jsonl`](sample_data.jsonl)(14개, 품질 위주 소규모 = LIMA 원칙)만 있으면
처음부터 끝까지 자체적으로 돌아갑니다. 두 가지 경로 중 하나를 고르세요. **둘 다 "Unsloth를 실제로
사용"** 합니다.

---

## 🅰️ Track A — Unsloth Studio (no-code, 과제 권장)

과제가 요구하는 **Unsloth Studio**(<http://unsloth.ai/docs/new/studio>)를 그대로 사용하는 최소 실습.
화면 단계 설명은 [`../../wiki/04-unsloth-studio-workflow.md`](../../wiki/04-unsloth-studio-workflow.md).

| 단계 | Studio에서 할 일 | 이 실습에서 |
|---|---|---|
| ① 모델 로드 | 베이스 모델 선택 | `Llama-3.2-1B-Instruct`(가장 가벼움) |
| ② 데이터 임포트 | 파일 업로드 | **`sample_data.jsonl` 업로드** |
| ③ Data Recipes | 형식 정제·미리보기 | role/chat template 정렬 확인 |
| ④ 학습 | 프리셋 QLoRA로 시작 | lr 2e-4 · r 16 · 1~3 epoch (loss 관찰) |
| ⑤ Model Arena | 베이스 vs 파인튜닝 비교 | 아래 "비교 프롬프트"로 나란히 비교 |
| ⑥ Export | GGUF 내보내기 | `q4_k_m` 선택 → Ollama로 로컬 실행 |

**⑤ 비교 프롬프트(학습에 쓴 주제 + 안 쓴 주제 섞기):**
- "QLoRA는 LoRA랑 뭐가 달라?" (학습 주제 → 우리 말투로 답하는지)
- "너는 누구야?" (페르소나 주입 확인)
- "오늘 서울 날씨 어때?" (무관한 질문 → 원래 능력 유지=망각 체크)

→ "베이스는 장황한데 파인튜닝은 우리 데이터 말투로 간결해졌다" 같은 차이를 캡처해 리포트에 사용.

---

## 🅱️ Track B — 코드로 (Google Colab, 무료 T4)

Studio가 내부에서 하는 일을 ~40줄 코드로 직접 보고 제어합니다.
파일: [`minimal_finetune.py`](minimal_finetune.py) (현재 Unsloth API 기준, 구문 검증 완료).

> ⚠️ **NVIDIA GPU(CUDA) 필수 — Mac/CPU에서는 실행되지 않습니다.** 무료 Colab을 쓰세요.

**Colab 실행 순서**
1. <https://colab.research.google.com> → 새 노트북 → 메뉴 *런타임 > 런타임 유형 변경 > T4 GPU*.
2. 셀에 설치: `!pip install unsloth`
3. `minimal_finetune.py`와 `sample_data.jsonl`을 업로드(좌측 파일 패널) 후:
   `!python minimal_finetune.py`  — 또는 셀에 코드를 붙여 단계별 실행(권장: loss를 눈으로 봄).

**코드 ↔ Studio 단계 대응** (스크립트 주석에 ①~⑥ 표시):
- ① `FastLanguageModel.from_pretrained(load_in_4bit=True)` → QLoRA 베이스 로드
- LoRA `get_peft_model(r=16, target_modules=[...All...])`
- ②③ `get_chat_template("llama-3.1")` + `sample_data.jsonl` 매핑
- ④ `SFTTrainer` + `SFTConfig(learning_rate=2e-4, ...)` → `trainer.train()`
- ⑤ `FastLanguageModel.for_inference` + `model.generate(...)`
- ⑥ `save_pretrained` / `save_pretrained_gguf(quantization_method="q4_k_m")`

---

## 🅲 Track C — 로컬 실행(Apple Silicon/CPU)으로 *실제로* 돌려보기

Unsloth는 **CUDA 전용**이라 Mac에선 못 돕니다. 그래서 **같은 LoRA 워크플로우**를 표준 스택
(transformers + PEFT)으로 옮긴 [`local_mps_finetune.py`](local_mps_finetune.py)를 두었습니다 —
이 맥(M-시리즈 GPU=MPS)에서 끝까지 실행됩니다. 학습되는 수학은 동일하고, Unsloth는 그 위의
속도·메모리 최적화 래퍼입니다.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch transformers peft accelerate datasets
cd tutorial/unsloth && python local_mps_finetune.py
```

### ✅ 실제 실행 결과 (Apple M4 Pro / MPS, 약 30초)

`Qwen2.5-0.5B-Instruct` + LoRA(r=16, 학습 파라미터 **1.75%**), `sample_data.jsonl` 14개로 20 epoch.
**loss 3.43 → 0.05.** 전체 로그: [`run_result.md`](run_result.md).

| 질문 | 학습 전 (베이스) | 학습 후 (LoRA) |
|---|---|---|
| QLoRA는 LoRA랑 뭐가 달라? | "QLoRA…LoRa技术的缩写…物联网…"(무선통신 LoRa와 혼동, 중국어) | **"얼린 원본을 4비트(NF4)로 양자화해 메모리를 약 75% 줄이고, 그 위에 LoRA 어댑터만 학습…"** |
| 너는 누구야? | "저는 AI 어시스턴트인 Qwen입니다…"(일반) | **"연세대 자연어정보분석 강의의 파인튜닝 실습을 돕는 조교 봇입니다…"** |

→ 우리 데이터의 **내용·페르소나·말투를 그대로 학습**. 단 14개·20 epoch라 과적합(=암기)에 가깝고,
이는 [02-hyperparameters](../../wiki/02-hyperparameters.md)의 "epoch↑ → 과적합" 그래프를 실제로 본 셈.
실전에선 데이터를 늘리고 epoch을 1~3으로 줄입니다.

> Track C는 "실제로 도는 것"을 보여주기 위한 로컬 대체물입니다. 과제 제출은 **Track A(Studio)** 또는
> **Track B(Colab+Unsloth)** 로 하세요 — 그게 과제가 요구하는 Unsloth입니다.

---

## 자신의 프로젝트로 바꾸기

1. `sample_data.jsonl`을 **자신의 도메인 데이터**로 교체(형식은 동일: `messages` 배열).
   → 데이터 설계: [`../../wiki/03-dataset-design.md`](../../wiki/03-dataset-design.md)
2. 베이스 모델·하이퍼파라미터 조정 → [`../../wiki/02-hyperparameters.md`](../../wiki/02-hyperparameters.md)
3. 결과를 [`../../report/final-project-template.md`](../../report/final-project-template.md)에 정리.

> 개념이 헷갈리면(왜 r? 왜 4bit?) → 선택 보충 [`../concepts/`](../concepts/)에서 numpy로 1분 체험.
