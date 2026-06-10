"""
로컬 실행용 LoRA 파인튜닝 — Apple Silicon(MPS)/CPU에서 '실제로' 도는 버전.

왜 이 파일이 따로 있나:
  Unsloth(minimal_finetune.py)는 CUDA(NVIDIA) 전용이라 Mac에서 못 돕니다.
  그래서 '동일한 LoRA 파인튜닝 워크플로우'를 표준 스택(transformers + PEFT)으로 옮겨
  이 맥의 GPU(MPS)에서 끝까지 실행되게 한 것입니다. 개념·단계는 1:1로 같습니다.
  (속도/메모리 최적화가 Unsloth의 역할이고, 학습되는 수학은 동일합니다.)

같은 폴더 sample_data.jsonl(14개)로 자체 실행. 학습 전/후 응답 변화를 출력합니다.
"""
import json, os, pathlib, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

HERE = pathlib.Path(__file__).parent
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"      # 작은 다국어 instruct (한국어 가능)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[env] device={device}  torch={torch.__version__}")

# ── ① 모델 + 토크나이저 로드 (Studio: 모델 로드) ─────────────────────────
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(device)

# ── LoRA 부착 (Studio: rank/alpha/target) — wiki/02 ───────────────────────
lora = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# ── ②③ 데이터 + chat template 정제 (Studio: 데이터/Data Recipes) — wiki/03 ─
rows = [json.loads(l) for l in open(HERE/"sample_data.jsonl") if l.strip()]
def encode(msgs):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    ids = tok(text, truncation=True, max_length=512)["input_ids"]
    return {"input_ids": ids, "labels": list(ids)}
train_ds = [encode(r["messages"]) for r in rows]
print(f"[data] {len(train_ds)}개 샘플 인코딩 완료")

def collate(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id
    ii, ll, am = [], [], []
    for b in batch:
        n = maxlen - len(b["input_ids"])
        ii.append(b["input_ids"] + [pad]*n)
        ll.append(b["labels"] + [-100]*n)        # pad 위치는 손실 제외
        am.append([1]*len(b["input_ids"]) + [0]*n)
    return {"input_ids": torch.tensor(ii), "labels": torch.tensor(ll),
            "attention_mask": torch.tensor(am)}

# ── 학습 전 응답 (베이스) ─────────────────────────────────────────────────
def ask(q, max_new=80):
    msgs = [{"role":"user","content":q}]
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    model.eval()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

PROMPTS = ["QLoRA는 LoRA랑 뭐가 달라?", "너는 누구야?"]
print("\n=== 학습 전 (베이스 모델) ===")
before = {q: ask(q) for q in PROMPTS}
for q,a in before.items(): print(f"  Q: {q}\n  A: {a}\n")

# ── ④ 학습 (Studio: 학습 — loss 관찰) — wiki/02 ───────────────────────────
args = TrainingArguments(
    output_dir=str(HERE/"outputs"),
    per_device_train_batch_size=2, gradient_accumulation_steps=2,
    num_train_epochs=20, learning_rate=2e-4,
    logging_steps=5, save_strategy="no", report_to=[], seed=3407,
    use_cpu=(device=="cpu"),
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collate)
print("=== 학습 시작 ===")
stats = trainer.train()
print(f"=== 학습 완료: 최종 train_loss = {stats.training_loss:.4f} ===\n")

# ── ⑤ 학습 후 응답 (Studio: Model Arena 비교) ────────────────────────────
print("=== 학습 후 (파인튜닝 모델) ===")
after = {q: ask(q) for q in PROMPTS}
for q,a in after.items(): print(f"  Q: {q}\n  A: {a}\n")

# ── ⑥ 어댑터 저장 (Studio: Export) ───────────────────────────────────────
save_dir = HERE/"lora_adapter"
model.save_pretrained(save_dir); tok.save_pretrained(save_dir)
print(f"[save] LoRA 어댑터 저장 → {save_dir}")

# 결과 로그 파일로도 남김
with open(HERE/"run_result.md","w") as f:
    f.write("# 로컬 LoRA 파인튜닝 실행 결과 (Apple M4 Pro / MPS)\n\n")
    f.write(f"- 모델: `{MODEL}`  · device: `{device}`\n")
    f.write(f"- 최종 train_loss: **{stats.training_loss:.4f}**  · 샘플 {len(train_ds)}개 · 20 epoch\n\n")
    for q in PROMPTS:
        f.write(f"### Q. {q}\n**학습 전:** {before[q]}\n\n**학습 후:** {after[q]}\n\n")
print("[save] run_result.md 작성 완료")
