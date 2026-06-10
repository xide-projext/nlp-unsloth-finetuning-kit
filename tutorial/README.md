# 실습 (tutorial)

과제의 핵심은 **직접 Unsloth를 돌려 보는 것**입니다. 그래서 실습은 두 층으로 나뉩니다.

## 🎯 [`unsloth/`](unsloth/) — 핵심 실습 (실제 파인튜닝)
**여기가 본 실습입니다.** 실제 Unsloth로 모델을 파인튜닝합니다.
- **Track A**: Unsloth **Studio**(no-code UI)로 6단계 — 과제 권장 경로.
- **Track B**: `minimal_finetune.py`로 같은 워크플로우를 코드로 (무료 Colab T4, 현재 API 검증 완료).
- 자체 포함 데이터 `sample_data.jsonl`(14개)로 처음부터 끝까지 돕니다.

→ 시작: [`unsloth/README.md`](unsloth/README.md)

## 🧩 [`concepts/`](concepts/) — 개념 보충 (선택)
"왜 r? 왜 4bit?"가 헷갈릴 때만, GPU 없이 numpy로 개념을 1분 체험.
**실습 대체용이 아닙니다.** → [`concepts/README.md`](concepts/README.md)

---

```
tutorial/
├── unsloth/      ← 핵심: 실제 Unsloth 파인튜닝 (Studio + Colab 코드 + 샘플 데이터)
└── concepts/     ← 선택: numpy 개념 데모 (rank/LoRA, 양자화, 토큰화)
```
