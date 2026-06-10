"""
실습 3) 토큰화 + 파인튜닝용 데이터셋 형식 — 표준 라이브러리만으로.

배우는 것:
  (a) BPE(Byte Pair Encoding)의 핵심: 자주 같이 나오는 글자쌍을 병합해 '서브워드' 만들기
  (b) 텍스트 → 토큰 → 정수 ID 변환 과정
  (c) Unsloth/instruction 파인튜닝이 요구하는 chat 형식(role 태깅) 데이터 만들기

이 실습은 데이터셋 설계가 왜 결과의 절반인지 감을 잡기 위한 것.

GPU 불필요. `python3 03_tokenization_and_dataset.py` 로 실행.
"""
import json
from collections import Counter

line = lambda c="-": print(c * 64)


def section(title):
    print()
    line("=")
    print(title)
    line("=")


# ──────────────────────────────────────────────────────────────
section("(a) BPE: 자주 붙어 다니는 쌍을 병합해 서브워드 만들기")
# 아주 작은 코퍼스로 BPE 학습 과정을 직접 보여준다
corpus = ["low", "low", "low", "low", "low",
          "lower", "lower",
          "newest", "newest", "newest", "newest", "newest", "newest",
          "widest", "widest", "widest"]

# 각 단어를 글자 단위 + 끝표시 </w> 로 시작
words = {tuple(list(w) + ["</w>"]): c for w, c in Counter(corpus).items()}


def get_pairs(words):
    pairs = Counter()
    for symbols, freq in words.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge(words, pair):
    new = {}
    for symbols, freq in words.items():
        s, out = list(symbols), []
        i = 0
        while i < len(s):
            if i < len(s) - 1 and (s[i], s[i + 1]) == pair:
                out.append(s[i] + s[i + 1]); i += 2
            else:
                out.append(s[i]); i += 1
        new[tuple(out)] = freq
    return new


print("시작(글자 단위):", list(words.keys())[0], "...")
for step in range(1, 6):
    pairs = get_pairs(words)
    best = pairs.most_common(1)[0]
    words = merge(words, best[0])
    print(f"  병합 {step}: {best[0][0]!r}+{best[0][1]!r} → "
          f"{(best[0][0]+best[0][1])!r}  (빈도 {best[1]})")
print("결과 토큰 예:", list(words.keys())[0], "|", list(words.keys())[-1])
print("  → 'est', 'low' 같은 서브워드가 생긴다. 희귀어도 조각 조합으로 표현 가능.")


# ──────────────────────────────────────────────────────────────
section("(b) 텍스트 → 토큰 → 정수 ID")
sentence = "자연어 처리 자연어 분석"
toks = sentence.split()           # (실제로는 서브워드지만 여기선 단어 단위로 단순화)
vocab = {t: i for i, t in enumerate(sorted(set(toks)))}
ids = [vocab[t] for t in toks]
print("문장 :", sentence)
print("토큰 :", toks)
print("어휘집(vocab):", vocab)
print("정수 ID:", ids, "  ← 모델은 글자가 아니라 이 숫자들을 먹는다")


# ──────────────────────────────────────────────────────────────
section("(c) 파인튜닝용 chat 형식 데이터셋 (role 태깅)")
# Unsloth/instruction 튜닝이 요구하는 표준 형태: messages = [{role, content}, ...]
dataset = [
    {"messages": [
        {"role": "user", "content": "연세대 자연어정보분석 강의가 뭐야?"},
        {"role": "assistant", "content": "NLP 이론과 LLM 파인튜닝 실습을 다루는 강의입니다."},
    ]},
    {"messages": [
        {"role": "user", "content": "LoRA가 뭐야?"},
        {"role": "assistant", "content": "원본 가중치는 얼리고 얇은 두 행렬만 학습하는 경량 파인튜닝 기법입니다."},
    ]},
]
print("JSONL 한 줄(=학습 샘플 1개) 예시:")
print(json.dumps(dataset[0], ensure_ascii=False, indent=2))

# chat template로 펼치면 모델이 실제로 보는 문자열 (형식은 모델마다 다름)
print("\nchat template 적용 시 모델이 보는 문자열(예시):")
for ex in dataset[:1]:
    s = ""
    for m in ex["messages"]:
        s += f"<|{m['role']}|>\n{m['content']}\n"
    s += "<|end|>"
    print(s)

print("핵심 체크리스트:")
print("  1) 모든 샘플이 토큰화 가능한 텍스트인가")
print("  2) user / assistant role이 정확히 태깅됐는가")
print("  3) 베이스 모델의 chat template과 형식이 맞는가")
print("  4) 품질(정확/일관된 답) > 양. 노이즈는 그대로 학습된다.")

print("\n끝. 핵심: 데이터 형식과 품질이 파인튜닝 결과의 절반 이상을 결정한다.")
