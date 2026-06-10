# 개념 보충 (선택) — numpy로 1분 직관

> ⚠️ 여기는 **선택적 보충**입니다. 과제의 핵심 실습은 [`../unsloth/`](../unsloth/)에서 **실제 Unsloth로
> 파인튜닝**하는 것입니다. 이 폴더는 "왜 r? 왜 4bit? rank가 뭐지?"가 헷갈릴 때, GPU 없이 numpy로
> 개념만 빠르게 체험하는 용도입니다. 이것으로 실습을 대신하지 마세요.

```bash
cd tutorial/concepts
python3 01_rank_and_lora.py
python3 02_quantization.py
python3 03_tokenization_and_dataset.py
```

| 스크립트 | 체험하는 개념 | 연결 위키 |
|---|---|---|
| `01_rank_and_lora.py` | rank=정보량, 저랭크 근사, LoRA r=0.39%, ΔW 학습 | [01](../../wiki/01-core-concepts.md) |
| `02_quantization.py` | 4-bit 양자화, 75% 절감, NF4식 오차 51%↓ | [01](../../wiki/01-core-concepts.md) |
| `03_tokenization_and_dataset.py` | BPE 병합, 토큰→ID, chat 데이터 형식 | [03](../../wiki/03-dataset-design.md) |

이 데모들은 **개념 증명용 축소 모델**이며 실제 학습이 아닙니다.
실제 파인튜닝 → [`../unsloth/README.md`](../unsloth/README.md).
