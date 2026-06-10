# 06. 용어집 (빠른 참조)

한 줄 정의 + 어디서 자세히 보는지. 자세한 비유·예시는 [`01`](01-core-concepts.md)·[`02`](02-hyperparameters.md).

| 용어 | 한 줄 정의 |
|---|---|
| **파인튜닝 (fine-tuning)** | 이미 학습된 모델을 특정 과제/도메인에 맞게 추가 학습. |
| **rank (랭크)** | 행렬이 담은 *독립적인 정보의 개수*. LoRA의 수학적 토대. |
| **저랭크 근사** | 큰 행렬을 얇은 두 행렬의 곱으로 근사(SVD/PCA 계열). |
| **LoRA** | 원본은 얼리고 변화량 ΔW를 `B·A`로 근사해 그 둘만 학습(~1%). |
| **ΔW (델타 W)** | 파인튜닝으로 생기는 가중치 변화량. LoRA가 이걸 저랭크로 근사. |
| **양자화 (quantization)** | 16-bit 실수를 4-bit 등 낮은 정밀도로 압축(매핑). |
| **NF4 (NormalFloat4)** | 가중치의 정규분포에 맞춰 눈금을 배치한 4-bit 타입. QLoRA가 사용. |
| **QLoRA** | 원본을 4-bit(NF4)로 얼리고 LoRA 어댑터만 학습. 메모리 ~75%↓. |
| **학습률 (learning rate, η)** | 한 번 업데이트의 *보폭*. LoRA 권장 2e-4. |
| **에폭 (epoch)** | 전체 데이터를 통독한 횟수. 권장 1–3. |
| **alpha (α)** | 어댑터 반영 강도. 적용은 `(α/r)·BA`. 흔히 α=2r. |
| **target modules** | LoRA를 붙일 트랜스포머 부품(q/k/v/o + gate/up/down). All 권장. |
| **batch size** | 한 업데이트에 보는 예시 수. |
| **gradient accumulation** | 작은 배치를 여러 번 모아 큰 배치 효과(메모리 절약). |
| **유효 배치 (effective batch)** | batch × grad-accum. 권장 16 내외. |
| **과적합 (overfitting)** | 학습 데이터에만 맞아 새 데이터에서 무너짐. |
| **과소적합 (underfitting)** | 충분히 못 배워 기본도 못 풂. |
| **training loss** | 학습 오차. 곡선으로 과적합/과소적합 진단. |
| **gradient norm** | 그래디언트 크기. 폭발하면 lr 과대 신호. |
| **토큰화 (tokenization)** | 텍스트를 정수 ID 조각으로 쪼개는 전처리. |
| **BPE** | 자주 붙는 글자쌍을 병합해 서브워드를 만드는 토큰화 방식. |
| **서브워드 (subword)** | 단어보다 잘게 쪼갠 토큰. 희귀어도 조합으로 표현. |
| **chat template** | 대화를 특수 토큰으로 감싸는 모델별 형식. 데이터가 여기 맞아야 함. |
| **role 태깅** | 각 발화를 user/assistant로 표시. |
| **instruct 모델** | 대화·지시 따르기를 이미 학습한 베이스 모델 계열. |
| **GGUF** | 로컬 실행(Ollama/llama.cpp)용 모델 배포 포맷. |
| **safetensors** | 가중치 저장 포맷(HuggingFace 표준). |
| **Model Arena** | Studio에서 베이스 vs 파인튜닝 출력을 비교하는 기능. |
| **Data Recipes** | PDF/CSV/DOCX 등을 학습 데이터로 자동 변환하는 Studio 기능. |
| **SFT** | Supervised Fine-Tuning. 입력-정답 쌍으로 하는 일반 파인튜닝. |
| **DPO/GRPO** | 선호·강화 기반 정렬 학습. lr이 SFT보다 훨씬 작음(예 5e-6). |
| **VRAM** | GPU 메모리. 모델·배치 크기를 제한하는 핵심 예산. |
| **OOM** | Out Of Memory. VRAM 초과 에러. batch↓·QLoRA로 대응. |

---

각 용어가 **어느 알고리즘 계보에서 나왔는지**는 [`07-lineage-map.md`](07-lineage-map.md),
대표 논문 색인은 [`05-resources.md`](05-resources.md) 참조.
