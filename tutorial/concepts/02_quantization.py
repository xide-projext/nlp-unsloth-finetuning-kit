"""
실습 2) 양자화(Quantization) 직관 — numpy만으로.

배우는 것:
  (a) 16-bit 실수를 4-bit(16칸 정수)로 압축하는 absmax 선형 양자화
  (b) 압축에 따른 오차 측정, 메모리 절감률
  (c) 신경망 가중치는 정규분포 → '균등 눈금'은 손해. NF4식 '분위수 눈금'이 왜 더 좋은지 비교

QLoRA = (가중치를 4-bit로 양자화해 얼림) + (LoRA 어댑터만 학습).
이 실습은 그중 '4-bit로 얼리는' 부분의 원리를 보여준다.

GPU 불필요. `python3 02_quantization.py` 로 실행.
"""
import numpy as np

np.random.seed(0)
line = lambda c="-": print(c * 64)


def section(title):
    print()
    line("=")
    print(title)
    line("=")


def absmax_quantize(x, bits=4):
    """대칭 absmax 선형 양자화: 범위를 [-max, max]로 보고 균등 칸에 매핑."""
    qmax = 2 ** (bits - 1) - 1            # 4-bit → 7
    scale = np.abs(x).max() / qmax
    q = np.round(x / scale).astype(int)  # 정수 코드 (-7..7)
    deq = q * scale                      # 복원값
    return q, deq, scale


# ──────────────────────────────────────────────────────────────
section("(a) 작은 벡터를 4-bit로 양자화해 보기")
x = np.array([-0.92, -0.13, 0.05, 0.21, 0.47, 0.88, 0.02, -0.55])
q, deq, scale = absmax_quantize(x, bits=4)
print("원본 (float):", np.round(x, 3))
print("정수 코드(4bit, -7..7):", q)
print("복원값:", np.round(deq, 3))
print(f"scale(눈금 간격) = {scale:.4f}")
print(f"평균 절대오차 = {np.abs(x-deq).mean():.4f}  ← 양자화로 잃은 정밀도")


# ──────────────────────────────────────────────────────────────
section("(b) 메모리 절감: float16 → int4")
n = 7_000_000_000   # 7B 모델 가정
b16 = n * 16 / 8 / 1e9
b4 = n * 4 / 8 / 1e9
print(f"7B 파라미터 기준")
print(f"  16-bit 저장: {b16:6.1f} GB")
print(f"  4-bit  저장: {b4:6.1f} GB")
print(f"  절감률: {(1 - b4/b16)*100:.0f}%  → '약 75% 절약'의 근거")
print("  이래서 8GB VRAM에서도 3B급 모델 파인튜닝이 가능해진다.")


# ──────────────────────────────────────────────────────────────
section("(c) 균등 눈금 vs 분위수 눈금(NF4 아이디어)")
# 신경망 가중치는 0 근처에 몰린 정규분포
W = np.random.randn(100_000) * 0.1

# 방법 1: 균등(선형) 4-bit
_, deqW_lin, _ = absmax_quantize(W, bits=4)
err_lin = np.abs(W - deqW_lin).mean()

# 방법 2: 분위수 기반 — 데이터를 16개 '동일 인원' 구간으로 나누고
#         각 구간의 대표값을 '그 구간에 든 값들의 평균(centroid)'으로 둔다.
#         (값이 몰린 0 근처에 눈금이 촘촘해진다 = NF4의 핵심 아이디어)
levels = 16
edges = np.quantile(W, np.linspace(0, 1, levels + 1))
idx = np.clip(np.digitize(W, edges[1:-1]), 0, levels - 1)
centers = np.array([W[idx == b].mean() for b in range(levels)])   # 구간 평균 = MSE 최적 대표값
deqW_q = centers[idx]
err_q = np.abs(W - deqW_q).mean()

print("가중치 분포: 평균 0, 0 근처에 집중 (정규분포)")
print(f"  균등 4-bit 눈금   → 평균 절대오차 {err_lin:.5f}")
print(f"  분위수 4-bit 눈금 → 평균 절대오차 {err_q:.5f}  (NF4식)")
print(f"  개선: {(1 - err_q/err_lin)*100:.0f}% 오차 감소")
print("  → 값이 몰린 곳에 눈금을 촘촘히 두면 같은 4-bit로 더 정확하다.")
print("    QLoRA가 NF4(NormalFloat4) 데이터타입을 쓰는 이유.")

print("\n끝. 핵심: 양자화는 'JPEG 압축'처럼 거의 안 보이게 용량을 줄이는 일.")
