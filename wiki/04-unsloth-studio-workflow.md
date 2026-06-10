# 04. Unsloth Studio 워크플로우 (단계별)

출처: <http://unsloth.ai/docs/new/studio>. Studio는 코드 없이(no-code) 파인튜닝을 돌리는 UI입니다.
공식 워크플로우를 그대로 따르면 됩니다.

```
① 모델 로드 → ② 데이터 임포트 → ③ Data Recipes 정제 → ④ 학습 → ⑤ Model Arena 비교 → ⑥ Export
```

---

## ① 베이스 모델 로드

**무엇을.** 파인튜닝의 출발점이 될 모델을 고릅니다("자신의 모델을 기준으로 잡고"가 과제의 핵심).

**선택 가이드 (입문 기준):**
- **8B급 instruct 모델**로 시작하는 걸 공식 문서도 권장 (이미 대화 형식을 학습한 -instruct 계열).
- VRAM이 빠듯하면 더 작은 3B급 + **QLoRA**.
- 모델별 세부 가이드(예: Qwen 계열)는 베이스 모델별 문서를 추가로 참조.

**리포트 포인트.** "왜 이 모델인가" — 크기(VRAM 예산), instruct 여부, 라이선스, 목표 도메인 적합성.

---

## ② 데이터 임포트

PDF·CSV·JSON·JSONL 등을 올립니다. 형식 요건은 [`03-dataset-design.md`](03-dataset-design.md) 참조.
JSONL `messages` 형식이 가장 안전합니다.

---

## ③ Data Recipes로 정제

업로드한 원본(특히 PDF/DOCX/TXT)을 학습 가능한 Q&A/chat 형식으로 자동 변환·정제.
- role 태깅이 맞는지, chat template과 정렬되는지 여기서 확인.
- 결과 미리보기로 **깨진 샘플·이상한 답변**을 걸러내세요.

---

## ④ 학습 (프리셋 또는 직접 설정)

**프리셋**으로 시작해도 되고, 직접 손잡이를 잡아도 됩니다. 권장 출발값:

| 손잡이 | 값 | 근거 |
|---|---|---|
| 방법 | **QLoRA**(VRAM 빠듯) / LoRA | [01-core-concepts](01-core-concepts.md) |
| learning rate | **2e-4** | [02-hyperparameters](02-hyperparameters.md) |
| epochs | **1–3** | 과적합 방지 |
| LoRA r | **16** | 표현력·메모리 균형 |
| LoRA α | **32** (=2r) | 경험칙 |
| target modules | **All** | attn+MLP 전체 |
| effective batch | **16** | batch×grad-accum |

**학습 중 모니터링.** Studio는 **loss·gradient norm·GPU 사용률**을 실시간으로 보여줍니다.
loss 곡선을 보며 과적합/과소적합을 판단 ([02 §loss 읽기](02-hyperparameters.md)). 스크린샷은
리포트 증거자료로 캡처해 두세요.

---

## ⑤ Model Arena — 베이스와 비교

파인튜닝한 모델과 **원본 베이스 모델**에 같은 프롬프트를 넣어 출력을 나란히 비교합니다.
"무엇이 어떻게 좋아졌는가"를 보여주는 가장 직접적인 증거 — 리포트의 핵심 자료.

**비교용 프롬프트 세트를 미리 만들어 두세요** (학습에 안 쓴 새 질문 5–10개). 도메인 적합 질문 +
일반 질문(원래 능력 유지 여부 = 망각 체크) 섞어서.

---

## ⑥ Export

학습 결과를 내보냅니다.
- **GGUF** — Ollama/llama.cpp 등 로컬 실행용 (양자화 옵션 선택 가능).
- **safetensors / merged 16-bit** — HuggingFace 업로드·추가 작업용.
- **LoRA 어댑터만** — 작은 용량(원본 + 어댑터 따로 배포).

내보낸 GGUF는 Ollama로 로컬에서 바로 돌려 데모할 수 있습니다(추천 영상의 export 단계 참고).

---

## 막혔을 때

- VRAM 부족(OOM) → 방법을 QLoRA로, batch↓ & grad-accum↑, 모델 크기↓, max sequence length↓.
- 출력이 안 변함 → epoch/r↑, 데이터 품질·형식 점검, target modules가 All인지 확인.
- 이상한 반복·붕괴 → lr↓, epoch↓(과적합), 데이터 중복 점검.

다음 → 결과를 [`report/final-project-template.md`](../report/final-project-template.md)에 정리.
