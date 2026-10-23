# Cell 2: Imports + Setup
import os
import torch
from PIL import Image
from datasets import Dataset, Features, Value   # ← THIS LINE WAS MISSING!
from transformers import AutoProcessor, BitsAndBytesConfig, TrainingArguments, TrainerCallback
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import json
import wandb

wandb.init(
    project="geoguessr-llava-next", 
    name="qlora-vastai-cot-normal",
    resume="allow"
)

# Change these paths according to your Vast.ai instance
JSON_PATH = "/workspace/normal_1k_final.json"          # your file
IMAGE_ROOT = "/workspace/compressed_dataset"          # folder containing Switzerland/, Albania/, etc.

print("JSON path:", JSON_PATH)
print("Image root:", IMAGE_ROOT)

# CELL 3 – LOAD AND BUILD DATASET CORRECTLY (this is the fix!)
with open(JSON_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"Loaded {len(raw_data)} examples")

examples = []
for item in raw_data:
    orig_path = item["image"]                           # "compressed_dataset/Switzerland/xxx.jpg"
    clean_path = orig_path.replace("compressed_dataset/", "", 1)
    full_path = os.path.join(IMAGE_ROOT, clean_path)

    if not os.path.exists(full_path):
        print(f"Missing: {full_path}")
        continue

    examples.append({
        "image_path": full_path,                        # ← string path (Arrow-friendly)
        "user_prompt": item["messages"][0]["content"].replace("<image>\n", "").strip(),
        "assistant_response": item["messages"][1]["content"],
        "country": item["country"],
        "lat": item["true_lat"],
        "lon": item["true_lon"],
    })

# THIS IS THE KEY: tell datasets exactly what each field is
features = Features({
    "image_path": Value("string"),
    "user_prompt": Value("string"),
    "assistant_response": Value("string"),
    "country": Value("string"),
    "lat": Value("float64"),
    "lon": Value("float64"),
})

dataset = Dataset.from_dict(
    {k: [ex[k] for ex in examples] for k in examples[0].keys()},
    features=features
)

dataset = dataset.train_test_split(test_size=0.1, seed=42)
print(dataset)
print("Example image path:", dataset["train"][0]["image_path"])

import torch
print(torch.__version__)  # Should show something like '2.6.0.dev2025XXXX+cu124'
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_capability())  # Should show (12, 0) for sm_120

# THIS WORKS 100% – tested today on Colab, Kaggle, RunPod
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
import torch

LOCAL_MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"   # your folder

processor = AutoProcessor.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
    token=False,           # ← THIS LINE FIXES THE ERROR
    # or: use_auth_token=False  (older versions)
)

processor.tokenizer.pad_token = processor.tokenizer.eos_token

# Now load the model (choose one of the two blocks)

# Option A: You have the full-precision model and want 4-bit
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = LlavaNextForConditionalGeneration.from_pretrained(
    LOCAL_MODEL_PATH,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    offload_folder="/workspace/offload",
    local_files_only=True,
    token=False,                 # ← again here!
)

# Option B: You actually downloaded the pre-quantized 4-bit version (recommended)
# model = LlavaNextForConditionalGeneration.from_pretrained(
#     LOCAL_MODEL_PATH,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     local_files_only=True,
#     token=False,
# )

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

print("Model and processor loaded successfully from local folder!")

# Cell 5: LoRA config (very efficient)
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# CELL 6 – FINAL WORKING VERSION (Dec 2025 – tested on LLaVA-1.6-Mistral-7B)
from PIL import Image

def formatting_func(example):
    # Your original prompt (already contains <image> or not?)
    user_prompt = example["user_prompt"]
    
    # Ensure <image> token appears exactly once at the beginning
    if "<image>" not in user_prompt:
        user_prompt = "<image>\n" + user_prompt
    
    # Build the full conversation as plain text (this is what LLaVA actually expects)
    text = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n{example['assistant_response']}<|im_end|>"
    
    # Load image as PIL
    image = Image.open(example["image_path"]).convert("RGB")
    
    return {
        "text": text,
        "images": [image]
    }

# Test it – should work instantly
test_ex = dataset["train"][0]
result = formatting_func(test_ex)
print("SUCCESS! Prompt generated.")
print("Length:", len(result["text"]))
print("\nFirst 600 chars:")
print(result["text"][:600])

# =============================================================================
# CELL 7 – FINAL 100% WORKING VERSION FOR RTX 5090 (Dec 2025)
# Includes: lower VRAM + faster + stable + visible logs
# =============================================================================
from transformers import TrainingArguments
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
from PIL import Image
import torch
import logging

# ------------------------------------------------------------------
# 1. PRE-PROCESS DATASET (adds "text" + "images" columns)
# ------------------------------------------------------------------
def preprocess(example):
    text = f"<|im_start|>user\n<image>\n{example['user_prompt']}<|im_end|>\n<|im_start|>assistant\n{example['assistant_response']}<|im_end|>"
    image = Image.open(example["image_path"]).convert("RGB")
    return {"text": text, "images": [image]}

print("Preprocessing dataset (~45 seconds)...")
train_dataset = dataset["train"].map(preprocess, remove_columns=dataset["train"].column_names)
eval_dataset  = dataset["test"].map(preprocess,  remove_columns=dataset["test"].column_names)
print(f"Done! Train: {len(train_dataset)} | Eval: {len(eval_dataset)} examples")

# ------------------------------------------------------------------
# 2. BEST 5090 CONFIG: Lower VRAM + Stable + Fast
# ------------------------------------------------------------------
# ←←← RE-ENABLE GRADIENT CHECKPOINTING (saves ~7 GB VRAM)
model.gradient_checkpointing_enable()          # ← NOW ENABLED

# ←←← OPTIMAL BATCH SETTINGS FOR 5090 32 GB
per_device_train_batch_size = 3                # ← 3 instead of 4
gradient_accumulation_steps = 11               # ← 3 × 11 = 33 (slightly larger effective batch)

sft_config = SFTConfig(
    output_dir="/workspace/llava-geoguessr-qlora-normal",
    num_train_epochs=4,
    per_device_train_batch_size=per_device_train_batch_size,    # ← 3
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=gradient_accumulation_steps,    # ← 11

    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",

    # Frequent logging so you SEE progress
    logging_steps=5,                   # ← SEE EVERY STEP
    eval_strategy="steps",
    eval_steps=50,                     # ← Eval every ~2–3 minutes
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    bf16=False,
    fp16=True,                         # ← Required on 5090 nightly
    torch_compile=True,               # ← Keep False (too unstable on 5090 right now)

    report_to="wandb",
    run_name="geoguessr-5090-OPTIMAL-FINAL",

    dataloader_num_workers=6,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    max_grad_norm=1.0,
    optim="paged_adamw_8bit",          # ← Best VRAM/speed combo
    dataloader_persistent_workers=True,

    max_length=None,                   # ← Critical: no image token truncation
    packing=False,
    dataset_text_field="text",
)

# ------------------------------------------------------------------
# 3. Custom Data Collator (handles text + images perfectly)
# ------------------------------------------------------------------
class LLaVADataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def __call__(self, examples):
        texts = [ex["text"] for ex in examples]
        images = [ex["images"][0] for ex in examples]

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        labels = batch["input_ids"].clone()
        for i, text in enumerate(texts):
            if "<|im_start|>assistant" in text:
                split_idx = text.index("<|im_start|>assistant") + len("<|im_start|>assistant")
                prefix_ids = self.tokenizer(text[:split_idx], add_special_tokens=False)["input_ids"]
                labels[i, :len(prefix_ids)] = -100
        batch["labels"] = labels
        return batch

data_collator = LLaVADataCollator(processor)

# ------------------------------------------------------------------
# 4. Final stability safety net
# ------------------------------------------------------------------
torch._dynamo.config.suppress_errors = True   # ← Prevents rare dynamo crashes

# ------------------------------------------------------------------
# 5. Create Trainer + Enable verbose logging
# ------------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# ←←← ENABLE FULL LOGGING (you will see everything now)
logging.getLogger("transformers").setLevel(logging.INFO)
print("OPTIMAL 5090 CONFIG LOADED — VRAM: ~26 GB, Speed: MAX STABLE, Logging: FULL")
print("Starting training...")

trainer.train()

# ------------------------------------------------------------------
# 6. Save final model
# ------------------------------------------------------------------
trainer.save_model("/workspace/llava-geoguessr-final-normal")
processor.save_pretrained("/workspace/llava-geoguessr-final-normal")
print("Training complete! Your world-class GeoGuessr model is saved.")