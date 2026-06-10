# 실습 (tutorial) — numpy만으로 핵심 개념 체험

GPU도, 추가 설치도 필요 없습니다. **numpy만** 있으면 됩니다(파이썬 3.x).
파인튜닝의 핵심 개념을 *손으로 돌려보며* 직관을 잡는 게 목적입니다.

```bash
cd tutorial
python3 01_rank_and_lora.py
python3 02_quantization.py
python3 03_tokenization_and_dataset.py
```

> 환경에 numpy가 없으면: `pip install numpy` (실습 3은 표준 라이브러리만 쓰므로 numpy 없이도 동작).

---

## 01_rank_and_lora.py — rank · 저랭크근사 · LoRA

- (a) 9칸짜리 행렬의 **rank가 1**임을 확인 → "정보의 개수" 감각.
- (b) 256×256 행렬을 **rank 8로 근사**하면 오차 0% — 본질 rank 도달.
- (c) 4096×4096에서 **LoRA r=8이 전체의 0.39%**임을 계산 → "1%만 학습" 근거.
- (d) `B=0`에서 시작(=원본 보존)해 경사하강으로 **목표 ΔW를 학습** → LoRA 학습 과정.

→ 개념: [`../wiki/01-core-concepts.md`](../wiki/01-core-concepts.md)

## 02_quantization.py — 4-bit 양자화 · NF4

- (a) 벡터를 **4-bit 정수(-7..7)로 양자화**하고 복원 오차 측정.
- (b) 7B 모델 16-bit(14GB) → 4-bit(3.5GB), **75% 절감** 계산.
- (c) 균등 눈금 vs **분위수 눈금(NF4식)** → 같은 4-bit로 **오차 51% 감소**.

→ 개념: [`../wiki/01-core-concepts.md`](../wiki/01-core-concepts.md) §양자화

## 03_tokenization_and_dataset.py — 토큰화 · 데이터 형식

- (a) **BPE 병합**을 직접 돌려 'est','low' 같은 서브워드가 생기는 과정.
- (b) 텍스트 → 토큰 → **정수 ID** 변환.
- (c) 파인튜닝용 **chat 형식(JSONL, role 태깅)** 데이터 만들기 + 체크리스트.

→ 개념: [`../wiki/03-dataset-design.md`](../wiki/03-dataset-design.md)

---

이 실습들은 **개념 증명용 축소 모델**입니다. 실제 파인튜닝은 Unsloth Studio에서
([`../wiki/04-unsloth-studio-workflow.md`](../wiki/04-unsloth-studio-workflow.md)).
