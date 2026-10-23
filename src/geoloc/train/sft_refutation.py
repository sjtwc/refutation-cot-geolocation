import os
import torch
from PIL import Image
from datasets import Dataset, Features, Value
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
import json
import wandb
import logging

wandb.init(project="geoguessr-llava-next", name="qlora-vastai-cot", resume="allow")

JSON_PATH = "/workspace/refutation_1k_final.json"
IMAGE_ROOT = "/workspace/compressed_dataset"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

examples = []
for item in raw_data:
    clean_path = item["image"].replace("compressed_dataset/", "", 1)
    full_path = os.path.join(IMAGE_ROOT, clean_path)
    if not os.path.exists(full_path):
        continue
    examples.append({
        "image_path": full_path,
        "user_prompt": item["messages"][0]["content"].replace("<image>\n", "").strip(),
        "assistant_response": item["messages"][1]["content"],
        "country": item["country"],
        "lat": item["true_lat"],
        "lon": item["true_lon"],
    })

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

LOCAL_MODEL_PATH = "llava-hf/llava-v1.6-mistral-7b-hf"

processor = AutoProcessor.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
    token=False,
)
processor.tokenizer.pad_token = processor.tokenizer.eos_token

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
    token=False,
)

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

def preprocess(example):
    text = f"<|im_start|>user\n<image>\n{example['user_prompt']}<|im_end|>\n<|im_start|>assistant\n{example['assistant_response']}<|im_end|>"
    image = Image.open(example["image_path"]).convert("RGB")
    return {"text": text, "images": [image]}

train_dataset = dataset["train"].map(preprocess, remove_columns=dataset["train"].column_names)
eval_dataset = dataset["test"].map(preprocess, remove_columns=dataset["test"].column_names)

per_device_train_batch_size = 3
gradient_accumulation_steps = 11

sft_config = SFTConfig(
    output_dir="/workspace/llava-geoguessr-qlora",
    num_train_epochs=4,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    report_to="wandb",
    run_name="geoguessr-5090-OPTIMAL-FINAL",
    dataloader_num_workers=6,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    max_grad_norm=1.0,
    optim="paged_adamw_8bit",
    dataloader_persistent_workers=True,
    dataset_text_field="text",
    packing=False,
)

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

torch._dynamo.config.suppress_errors = True

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

logging.getLogger("transformers").setLevel(logging.INFO)

trainer.train()

trainer.save_model("/workspace/llava-geoguessr-final")
processor.save_pretrained("/workspace/llava-geoguessr-final")