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
│   ├── 05-resources.md             · 추천 영상·문서 + 대표 논문 마스터 색인(12편)
│   ├── 06-glossary.md              · 한 줄 용어집 (빠른 참조)
│   ├── 07-lineage-map.md           · 계보 지도: 고전(LSA·word2vec)→LoRA/QLoRA 연결 (Mermaid)
│   └── diagrams/                   · 계보 다이어그램 PNG 6장(슬라이드용) + Mermaid 소스
├── tutorial/                       ← 실습
│   ├── unsloth/                    · 🎯 핵심: 실제 Unsloth 파인튜닝
│   │   ├── README.md               ·   Track A(Studio no-code) + Track B(Colab 코드)
│   │   ├── minimal_finetune.py     ·   최소 파인튜닝 코드 (현재 API 검증, T4 실행)
│   │   └── sample_data.jsonl       ·   자체 포함 샘플 데이터(14개)
│   └── concepts/                   · 🧩 선택: numpy 개념 데모 (rank/LoRA·양자화·토큰화)
└── report/
    └── final-project-template.md   ← 기말 리포트 작성 템플릿
```

---

## 🚀 빠른 시작 (3단계)

**1. 실제로 파인튜닝** — 과제 핵심. 무료 Colab(T4) 또는 Unsloth Studio.
```
tutorial/unsloth/README.md 를 따라
  · Track A: Unsloth Studio(no-code)로 sample_data.jsonl 업로드 → QLoRA 학습 → 비교 → GGUF export
  · Track B: Colab에서  !pip install unsloth  후  minimal_finetune.py  실행
```
> (개념이 헷갈리면 `tutorial/concepts/`에서 numpy로 1분 체험 — *선택 보충*.)

**2. 위키 읽기** — `wiki/00-overview.md` → `01` → `02` → `03` → `04` 순서 권장.

**3. 프로젝트 수행** — `wiki/04-unsloth-studio-workflow.md`를 따라 Studio에서 실행,
   결과는 `report/final-project-template.md`에 채워 넣기.

---

## 🎯 한 줄 요약

> **LoRA** = 큰 모델은 얼리고 *얇은 두 행렬만* 학습 (파라미터 ~1%).
> **QLoRA** = 거기에 원본을 *4-bit로 압축* (메모리 ~75%↓).
> 이걸 **Unsloth Studio**라는 no-code UI로 돌리는 게 이번 프로젝트.
> 점수는 "왜 이 모델·이 데이터·이 하이퍼파라미터를 골랐는가"를 **근거 있게** 쓰는 데서 갈린다.

자세한 출처와 권장값은 [`wiki/05-resources.md`](wiki/05-resources.md) 참조.
