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

## 🔬 세 가지 접근 비교 (이번 실제 실행 결과 기반)

같은 질문 *"QLoRA는 LoRA랑 뭐가 달라?"* 를 기준으로, 직접 돌려 본 결과로 비교합니다.
(전체 로그: [`tutorial/unsloth/run_result.md`](tutorial/unsloth/run_result.md))

| | **① 원 모델 그대로** (파인튜닝 X) | **② numpy 개념 데모** | **③ Unsloth 파인튜닝** (실제 학습) |
|---|---|---|---|
| **무엇을 하나** | 사전학습 모델에 그냥 질문(프롬프팅) | rank·LoRA·양자화·토큰화의 *수학*을 축소 모델로 체험 | 베이스에 LoRA/QLoRA로 *우리 데이터*를 학습 |
| **이번 실측 결과** | ❌ "QLoRA…LoRa技术…物联网"(무선통신 LoRa로 착각·중국어) | 📐 LoRA r=8→**0.39%**, 양자화 **75%↓**, NF4 오차 **51%↓** (개념 수치) | ✅ "얼린 원본을 4비트(NF4)로…**75%** 줄이고 어댑터만 학습" (loss 3.43→**0.05**, 30초) |
| **장점** | 학습 0·비용 0·즉시 사용, 범용 지식 풍부 | 설치·GPU 불필요, 초 단위, 내부 원리 100% 투명 | 도메인·페르소나·말투를 **모델에 내재화**, 파라미터 **1.75%만** 학습, 메모리 절약 |
| **단점** | 도메인·말투 미반영, 엉뚱한 답·환각 | 실제 언어모델 아님 → **텍스트 생성·학습 불가** | **GPU 필요**(Unsloth=CUDA 전용), 데이터 품질에 좌우, 과적합 위험 |
| **가능성** | 프롬프트 엔지니어링·RAG로 부분 보완 | 개념 검증·강의 자료·리포트 근거 그림 | GGUF export→**Ollama 로컬 배포**, 모델 교체·스케일업, DPO/GRPO 확장 |
| **한계** | 가중치 불변 → 새 지식·스타일 **내재화 못함**, 긴 프롬프트 비용 | 과제 제출물이 **될 수 없음**(개념용) | 사전학습에 없던 능력 *창조*는 못함(정렬·스타일 위주). Mac은 Unsloth 직접 불가→Colab/MPS 대체 |
| **이 키트 위치** | 파인튜닝의 *비교 기준선* (Model Arena의 베이스) | [`tutorial/concepts/`](tutorial/concepts/) (선택 보충) | [`tutorial/unsloth/`](tutorial/unsloth/) (🎯 핵심) |

> **언제 무엇을?** 빠른 답·범용 작업 → ①(+RAG). 개념 이해·발표 → ②. **도메인 특화·과제 제출 → ③**.
> ②와 ③은 경쟁이 아니라 *이해(②)→실행(③)* 의 순서입니다. ①은 ③의 향상을 측정하는 기준선입니다.

---

## 🎯 한 줄 요약

> **LoRA** = 큰 모델은 얼리고 *얇은 두 행렬만* 학습 (파라미터 ~1%).
> **QLoRA** = 거기에 원본을 *4-bit로 압축* (메모리 ~75%↓).
> 이걸 **Unsloth Studio**라는 no-code UI로 돌리는 게 이번 프로젝트.
> 점수는 "왜 이 모델·이 데이터·이 하이퍼파라미터를 골랐는가"를 **근거 있게** 쓰는 데서 갈린다.

자세한 출처와 권장값은 [`wiki/05-resources.md`](wiki/05-resources.md) 참조.
