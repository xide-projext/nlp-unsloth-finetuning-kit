# 00. 프로젝트 큰그림 & 진행 순서

## 이번 프로젝트가 진짜 묻는 것

과제는 "Unsloth Studio로 모델을 파인튜닝하라"지만, 점수는 **버튼을 눌렀다**가 아니라
**왜 그렇게 했는지 설명할 수 있다**에서 갈립니다. no-code UI 조작법보다 그 뒤에서
돌아가는 개념(LoRA·양자화·하이퍼파라미터)을 이해하는 게 결과물과 리포트 양쪽에 직결됩니다.

리포트에서 반드시 답할 수 있어야 하는 질문:

| 질문 | 어디서 근거를 얻나 |
|---|---|
| 왜 이 **베이스 모델**을 골랐나? | [04-workflow](04-unsloth-studio-workflow.md) §모델 선택 |
| LoRA냐 **QLoRA**냐, 왜? | [01-core-concepts](01-core-concepts.md) |
| 데이터셋을 어떻게 **설계**했나? | [03-dataset-design](03-dataset-design.md) |
| rank·alpha·lr·epoch을 **왜 그 값**으로? | [02-hyperparameters](02-hyperparameters.md) |
| **과적합/과소적합**을 어떻게 판단했나? | [02-hyperparameters](02-hyperparameters.md) §loss 읽기 |
| 베이스 대비 **얼마나 좋아졌나**? | Studio의 Model Arena 비교 |

---

## 4가지 필수 개념 (시작 전 반드시)

1. **LoRA / QLoRA의 원리** — 전체 파라미터 대신 얇은 두 행렬(약 1%)만 학습.
   LoRA는 원본을 16-bit로, QLoRA는 4-bit로 양자화해 메모리 ~75% 절약. → 둘 중 무엇을 쓸지 결정.
2. **데이터셋 설계** — 결과의 절반 이상이 여기서 갈림. role 태깅 + 모델 chat template 정렬.
3. **하이퍼파라미터 감각** — 무엇을 만지면 무엇이 변하는지. (lr 2e-4, epoch 1–3, r 4–64, α≥r …)
4. **과적합 vs 과소적합** — training loss를 보며 진단. Studio가 loss·gradient norm·GPU를 실시간 표시.

---

## 권장 진행 순서 (Unsloth Studio 공식 워크플로우)

```
① 베이스 모델 로드   →  ② 데이터 임포트(PDF/CSV/JSONL)  →  ③ Data Recipes로 정제
        ↓                                                            ↓
⑥ GGUF/safetensor export  ←  ⑤ Model Arena로 베이스와 비교  ←  ④ 프리셋/직접설정으로 학습
```

각 단계의 화면·설정은 [`04-unsloth-studio-workflow.md`](04-unsloth-studio-workflow.md)에 단계별로.

---

## 학습 로드맵 (이 키트 사용법)

| 순서 | 할 일 | 파일 |
|---|---|---|
| 1 | 손으로 직관 잡기 (10분) | `tutorial/*.py` 3개 실행 |
| 2 | 핵심 개념 정독 | [01-core-concepts](01-core-concepts.md) |
| 3 | 손잡이 이해 | [02-hyperparameters](02-hyperparameters.md) |
| 4 | 데이터셋 설계 | [03-dataset-design](03-dataset-design.md) |
| 5 | Studio 실행 | [04-workflow](04-unsloth-studio-workflow.md) |
| 6 | 리포트 작성 | [`report/final-project-template.md`](../report/final-project-template.md) |

막히면 [`06-glossary.md`](06-glossary.md)에서 용어 한 줄 정의를, 출처는 [`05-resources.md`](05-resources.md)에서.
