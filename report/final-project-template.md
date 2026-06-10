# 기말 프로젝트 리포트 — [제목]

> 자연어정보분석 · Unsloth 파인튜닝 프로젝트
> 작성자: ___  /  학번: ___  /  날짜: ___

채점은 "버튼을 눌렀다"가 아니라 **"왜 그렇게 했는지 근거 있게 설명한다"**에서 갈립니다.
각 절의 〔근거〕에는 위키/실습/공식 문서를 인용하세요.

---

## 1. 문제 정의 & 목표

- 무엇에 특화시키려 했나(도메인/말투/포맷/분류 등):
- 성공 기준(무엇이 어떻게 좋아지면 성공인가):

## 2. 베이스 모델 선택

| 항목 | 내용 |
|---|---|
| 모델명 / 크기 | |
| instruct 여부 | |
| 라이선스 | |
| VRAM 예산 / 실행 환경 | |

〔근거〕왜 이 모델인가 (크기 vs VRAM, 도메인 적합성 등):

## 3. 방법: LoRA vs QLoRA

- 선택: ☐ LoRA  ☐ QLoRA
- 〔근거〕(VRAM 예산, 품질·속도 트레이드오프 — [wiki/01](../wiki/01-core-concepts.md)):

## 4. 데이터셋 설계

| 항목 | 내용 |
|---|---|
| 출처 | |
| 형식 (JSONL/CSV …) | |
| 샘플 수 (train / valid) | |
| 정제 방법 (Data Recipes 등) | |

- 샘플 예시 1–2개(붙여넣기):
- 〔근거〕품질 체크리스트 통과 여부 — [wiki/03](../wiki/03-dataset-design.md):

## 5. 하이퍼파라미터

| 손잡이 | 사용값 | 〔근거〕왜 이 값 |
|---|---|---|
| learning rate | | |
| epochs | | |
| LoRA r | | |
| LoRA alpha | | |
| target modules | | |
| effective batch (batch×accum) | | |

근거 참조: [wiki/02](../wiki/02-hyperparameters.md).

## 6. 학습 과정 & 모니터링

- training loss 곡선 (스크린샷 첨부):
- gradient norm / GPU 사용률 관찰:
- 〔분석〕과적합/과소적합 판단과 그에 따라 무엇을 바꿨나:

## 7. 결과: 베이스 vs 파인튜닝 (Model Arena)

학습에 안 쓴 새 프롬프트로 비교. 도메인 질문 + 일반 질문(망각 체크) 섞기.

| 프롬프트 | 베이스 출력 | 파인튜닝 출력 | 평가 |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

- 〔분석〕무엇이 좋아졌나 / 부작용(망각·반복 등)은 없나:

## 8. Export & 데모

- 내보낸 포맷: ☐ GGUF ☐ safetensors ☐ LoRA 어댑터
- 실행 방법(예: Ollama로 로컬 데모):

## 9. 한계 & 다음 단계

- 아쉬운 점 / 데이터·하이퍼파라미터로 더 해볼 것:

## 10. 참고문헌

- Unsloth Studio 문서, Fine-tuning Guide, LoRA Hyperparameters Guide ([wiki/05](../wiki/05-resources.md))
- LoRA(arXiv:2106.09685), QLoRA(arXiv:2305.14314)
- (직접 본 영상·블로그 추가)

---

### 부록: 재현 정보
- 환경(GPU/Colab, VRAM):
- 학습 소요 시간:
- 랜덤 시드 / 기타 설정:
