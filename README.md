# 자연어정보분석 기말 프로젝트 — Unsloth 파인튜닝 학습 키트

연세대 **자연어정보분석** 강의 기말 프로젝트(Unsloth Studio로 베이스 모델을 직접 파인튜닝)를
충실히 수행하기 위한 **개념 위키 + 실행 가능한 실습 + 리포트 템플릿** 모음입니다.

> 프로젝트 과제: <http://unsloth.ai/docs/new/studio> 를 바탕으로, 자신의 베이스 모델을 잡고 파인튜닝.

---

## 📂 무엇이 들어있나

```
2026-06-nlp/
├── README.md                       ← 지금 이 파일 (전체 길잡이)
├── wiki/                           ← 개념 위키 (강의·리포트용, 개념→비유→예시 구조)
│   ├── 00-overview.md              · 프로젝트 큰그림 + 진행 순서 + 채점 대비
│   ├── 01-core-concepts.md         · rank / LoRA / QLoRA / 양자화 (핵심 개념)
│   ├── 02-hyperparameters.md       · 학습률·에폭·rank·alpha·batch (손잡이 가이드)
│   ├── 03-dataset-design.md        · 데이터셋 설계 (결과의 절반)
│   ├── 04-unsloth-studio-workflow.md · Studio 화면 단계별 워크플로우
│   ├── 05-resources.md             · 추천 영상·문서·링크 (출처 정리)
│   └── 06-glossary.md              · 한 줄 용어집 (빠른 참조)
├── tutorial/                       ← numpy만으로 돌아가는 개념 실습 (GPU 불필요)
│   ├── README.md
│   ├── 01_rank_and_lora.py         · rank·저랭크근사·LoRA 학습 체험
│   ├── 02_quantization.py          · 4-bit 양자화·NF4 직관
│   └── 03_tokenization_and_dataset.py · BPE·토큰화·chat 데이터 형식
└── report/
    └── final-project-template.md   ← 기말 리포트 작성 템플릿
```

---

## 🚀 빠른 시작 (3단계)

**1. 개념 잡기** — 손으로 돌려보며 직관부터 (10분, 설치 불필요)
```bash
cd tutorial
python3 01_rank_and_lora.py        # LoRA가 왜 파라미터 1%만 학습하는지
python3 02_quantization.py         # QLoRA가 왜 메모리 75% 아끼는지
python3 03_tokenization_and_dataset.py  # 데이터셋을 어떤 형식으로 만드는지
```

**2. 위키 읽기** — `wiki/00-overview.md` → `01` → `02` → `03` 순서 권장.

**3. 프로젝트 수행** — `wiki/04-unsloth-studio-workflow.md`를 따라 Studio에서 실행,
   결과는 `report/final-project-template.md`에 채워 넣기.

---

## 🎯 한 줄 요약

> **LoRA** = 큰 모델은 얼리고 *얇은 두 행렬만* 학습 (파라미터 ~1%).
> **QLoRA** = 거기에 원본을 *4-bit로 압축* (메모리 ~75%↓).
> 이걸 **Unsloth Studio**라는 no-code UI로 돌리는 게 이번 프로젝트.
> 점수는 "왜 이 모델·이 데이터·이 하이퍼파라미터를 골랐는가"를 **근거 있게** 쓰는 데서 갈린다.

자세한 출처와 권장값은 [`wiki/05-resources.md`](wiki/05-resources.md) 참조.
