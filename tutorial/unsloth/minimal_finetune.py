"""
최소 Unsloth 파인튜닝 — Unsloth Studio의 6단계를 '코드'로 그대로 (Track B).

이 스크립트 = Studio가 내부에서 하는 일과 동일한 워크플로우입니다.
  ① 모델 로드 → ② 데이터 → ③ 정제(chat template) → ④ 학습 → ⑤ 비교(추론) → ⑥ export(GGUF)

⚠️ 실행 환경: NVIDIA GPU(CUDA)가 필요합니다. **Mac/CPU에서는 돌지 않습니다.**
   → 무료 Google Colab(T4 GPU)에서 실행하세요. (README.md의 Colab 안내 참조)
   같은 폴더의 sample_data.jsonl(14개 샘플)만 있으면 자체적으로 끝까지 돕니다.

설치(Colab):  !pip install unsloth
API 기준: unslothai/notebooks의 Llama3.2 Conversational (2026) 노트북.
"""

# ─────────────────────────────────────────────────────────────────────────
# ① 베이스 모델 로드  (Studio: "모델 로드")
# ─────────────────────────────────────────────────────────────────────────
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None            # None=자동 (T4→float16, Ampere+→bfloat16)
load_in_4bit = True     # True = QLoRA(4bit). VRAM 빠듯하면 그대로 둔다.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",  # 가장 작은 입문용. 3B면 "...-3B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...",  # gated 모델일 때만
)

# ─────────────────────────────────────────────────────────────────────────
#   LoRA 어댑터 부착  (Studio: 학습 설정의 rank/alpha/target)
#   → 하이퍼파라미터 근거: wiki/02-hyperparameters.md
# ─────────────────────────────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                              # rank: 8/16/32… (작은 모델은 16에서 시작)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],   # All (attn+MLP)
    lora_alpha = 16,                     # 흔히 r 또는 2r
    lora_dropout = 0,                    # 0이 최적화됨
    bias = "none",
    use_gradient_checkpointing = "unsloth",   # 긴 컨텍스트·메모리 절약
    random_state = 3407,
    use_rslora = False,                  # True면 alpha/√r (큰 rank에 안정) — wiki/02 참조
)

# ─────────────────────────────────────────────────────────────────────────
# ② 데이터 임포트 + ③ chat template 정제  (Studio: "데이터 임포트 / Data Recipes")
#   → 데이터 설계 근거: wiki/03-dataset-design.md
# ─────────────────────────────────────────────────────────────────────────
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset

tokenizer = get_chat_template(tokenizer, chat_template = "llama-3.1")

def formatting_prompts_func(examples):
    # sample_data.jsonl은 {"messages":[{role,content}...]} 형식
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(c, tokenize = False, add_generation_prompt = False)
             for c in convos]
    return {"text": texts}

dataset = load_dataset("json", data_files = "sample_data.jsonl", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True)
print(f"학습 샘플 수: {len(dataset)}")
print("정제된 첫 샘플(모델이 보는 문자열):\n", dataset[0]["text"][:300])

# ─────────────────────────────────────────────────────────────────────────
# ④ 학습  (Studio: "학습" — loss를 실시간 모니터링하는 그 단계)
# ─────────────────────────────────────────────────────────────────────────
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,     # 유효 배치 = 2×4 = 8
        warmup_steps = 5,
        max_steps = 30,                      # 데모용. 실전은 num_train_epochs=1~3, max_steps=None
        # num_train_epochs = 1,
        learning_rate = 2e-4,                # wiki/02 권장 출발값
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

# (선택) 응답 부분에만 손실을 걸어 학습 품질↑ — Llama-3.1 태그 기준
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

trainer_stats = trainer.train()       # ← loss가 줄어드는지 관찰 (과적합/과소적합 판단)
print("학습 완료:", trainer_stats.metrics)

# ─────────────────────────────────────────────────────────────────────────
# ⑤ 추론으로 베이스와 비교  (Studio: "Model Arena")
# ─────────────────────────────────────────────────────────────────────────
FastLanguageModel.for_inference(model)   # 2x 빠른 추론
messages = [{"role": "user", "content": "QLoRA는 LoRA랑 뭐가 달라?"}]   # 학습에 쓴 주제
inputs = tokenizer.apply_chat_template(
    messages, tokenize = True, add_generation_prompt = True, return_tensors = "pt"
).to("cuda")
outputs = model.generate(input_ids = inputs, max_new_tokens = 128, use_cache = True,
                         temperature = 0.7)
print("파인튜닝 모델 응답:\n", tokenizer.batch_decode(outputs)[0])

# ─────────────────────────────────────────────────────────────────────────
# ⑥ Export  (Studio: "Export")
# ─────────────────────────────────────────────────────────────────────────
# (a) LoRA 어댑터만 저장 (가장 가벼움)
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# (b) GGUF로 export → Ollama/llama.cpp 로컬 실행용 (필요할 때 True)
if False:
    model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method = "q4_k_m")
# (c) HF 허브 업로드 (토큰 필요)
if False:
    model.push_to_hub_gguf("HF_USERNAME/my_finetune", tokenizer,
                           quantization_method = "q4_k_m", token = "hf_...")

print("\n끝. 이 흐름이 Unsloth Studio의 6단계와 1:1로 대응합니다 → wiki/04-unsloth-studio-workflow.md")
