"""
실습 1) 행렬 rank와 LoRA의 핵심 직관 — numpy만으로.

배우는 것:
  (a) 행렬의 rank = "독립적인 정보의 개수"라는 감각
  (b) 저랭크 근사(low-rank approximation)로 큰 행렬을 두 개의 얇은 행렬로 표현
  (c) LoRA가 절약하는 파라미터 수를 직접 계산
  (d) 작은 LoRA(B·A)를 경사하강으로 학습시켜 목표 변화량 ΔW를 따라가게 하기

GPU 불필요. `python3 01_rank_and_lora.py` 로 실행.
"""
import numpy as np

np.random.seed(0)
np.seterr(all="ignore")   # Python 3.14 + numpy 2.2.6 BLAS의 잘못된 matmul 경고 억제(결과값은 정확)
line = lambda c="-": print(c * 64)


def section(title):
    print()
    line("=")
    print(title)
    line("=")


# ──────────────────────────────────────────────────────────────
section("(a) rank = 독립적인 정보의 개수")
# 둘째 행 = 첫째 행 x2, 셋째 행 = 첫째 행 x3  →  독립적인 줄은 1개뿐
A = np.array([[1.0, 2.0, 3.0],
              [2.0, 4.0, 6.0],
              [3.0, 6.0, 9.0]])
print("행렬 A =\n", A)
print(f"칸(원소)은 9개지만, 실제 rank = {np.linalg.matrix_rank(A)}  → 정보는 한 줄 분량")

B = np.array([[1.0, 0.0, 1.0],
              [0.0, 1.0, 1.0],
              [1.0, 1.0, 0.0]])
print(f"\n서로 섞이지 않는 B 의 rank = {np.linalg.matrix_rank(B)}  → 꽉 찬 정보")


# ──────────────────────────────────────────────────────────────
section("(b) 저랭크 근사: 큰 행렬 ≈ 얇은 두 행렬의 곱")
# 일부러 '본질적으로 낮은 rank'인 큰 행렬을 만든다: (256x8) @ (8x256)
d, k, true_r = 256, 256, 8
W = np.random.randn(d, true_r) @ np.random.randn(true_r, k)
print(f"W 크기 = {d}x{k} = {d*k:,} 개 원소,  하지만 본질 rank = {np.linalg.matrix_rank(W)}")

# SVD로 rank r 근사를 만들어 본다
U, S, Vt = np.linalg.svd(W)
for r in [1, 4, 8, 16]:
    W_approx = (U[:, :r] * S[:r]) @ Vt[:r, :]
    err = np.linalg.norm(W - W_approx) / np.linalg.norm(W)
    params = d * r + r * k
    print(f"  rank {r:>2} 근사 → 저장 파라미터 {params:>6,}개 "
          f"({params/(d*k)*100:5.1f}%),  상대오차 {err:6.2%}")
print("  → 본질 rank(8)에 도달하면 오차가 사실상 0. 그 이상은 낭비.")


# ──────────────────────────────────────────────────────────────
section("(c) LoRA 파라미터 절약: h = Wx + (B·A)x")
# 트랜스포머의 흔한 가중치 크기로 계산
for dim in [4096]:
    full = dim * dim
    print(f"가중치 {dim}x{dim} = {full:,}개 (전체 파인튜닝이면 전부 학습)")
    for r in [8, 16, 32, 64]:
        lora = dim * r + r * dim          # A:(r x dim), B:(dim x r)
        print(f"  LoRA r={r:>2}: 학습 파라미터 {lora:>9,}개  =  전체의 {lora/full*100:5.2f}%")
    print("  → '전체 가중치의 약 1%만 최적화' 라는 말의 근거.")


# ──────────────────────────────────────────────────────────────
section("(d) LoRA 어댑터(B·A)를 직접 학습시켜 보기")
# 목표: 어떤 '변화량' ΔW(저랭크)를 B·A 로 따라가게 만든다.
d, k, r = 64, 64, 4
delta_true = np.random.randn(d, r) @ np.random.randn(r, k) * 0.1   # 목표 ΔW (rank 4)
X = np.random.randn(k, 256)                                        # 입력 배치
Y = delta_true @ X                                                 # 목표 출력

# LoRA 초기화: B=0 (그래서 시작 시 모델은 원본과 동일), A는 랜덤
A = np.random.randn(r, k) * 0.01
Bm = np.zeros((d, r))
lr, alpha = 0.05, 8
scale = alpha / r           # LoRA 스케일 (alpha/r)

print(f"목표 ΔW: rank={np.linalg.matrix_rank(delta_true)}, "
      f"LoRA r={r}, scale=alpha/r={scale}")
for step in range(0, 401):
    pred = scale * (Bm @ (A @ X))
    err = pred - Y
    loss = np.mean(err ** 2)
    # gradient (체인룰)
    gB = scale * (err @ X.T @ A.T) / X.shape[1]
    gA = scale * (Bm.T @ err @ X.T) / X.shape[1]
    Bm -= lr * gB
    A -= lr * gA
    if step % 80 == 0:
        recon = np.linalg.norm(scale * Bm @ A - delta_true) / np.linalg.norm(delta_true)
        print(f"  step {step:>3} | MSE loss {loss:.5f} | ΔW 복원 상대오차 {recon:6.2%}")
print("  → B=0에서 시작해(=원본 보존) 점점 목표 변화량을 학습. 이것이 LoRA 학습.")

print("\n끝. 핵심: 큰 변화도 '얇은 두 행렬'로 충분히 표현된다는 게 LoRA의 전제.")
