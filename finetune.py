import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import json
import random
import time
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# === CONFIG ===
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
OUTPUT_DIR = "./finetuned_model"
DATA_FILE = "training_data.json"

SYSTEM_PROMPT = """Sen uzman bir yazılım mühendisisin. Adın "Code Assistant".
<<<<<<< HEAD
=======
<<<<<<< HEAD
SADECE TÜRKÇE cevap ver. Asla Çince, Japonca veya başka bir dil kullanma.
=======
>>>>>>> b893943f309bd34c665722ceb6077ca77d160779
>>>>>>> b58b1ff (chore: remove large model files and update gitignore)
Türkçe açıklama yap, kod İngilizce olsun.
Çalışan, test edilebilir kod ver.
Neden böyle yazdığını açıkla.
Best practice ve clean code prensiplerine uy."""

# === LOAD MODEL (4-bit quantization ile VRAM tasarrufu) ===
print("Model yükleniyor (4-bit quantization)...")
print(f"({MODEL_NAME})")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === LoRA CONFIG (VRAM'e sığacak boyut) ===
print("\nLoRA konfigürasyonu...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === LOAD DATA ===
print("\nVeri yükleniyor...")
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# Veriyi çoğalt (4 saat eğitim için)
random.seed(42)
augmented_data = []
for _ in range(80):
    shuffled = raw_data.copy()
    random.shuffle(shuffled)
    augmented_data.extend(shuffled)

print(f"Orijinal örnek: {len(raw_data)}")
print(f"Augmented örnek: {len(augmented_data)}")

# === FORMAT DATA ===
def format_example(example):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}

formatted = [format_example(ex) for ex in augmented_data]
dataset = Dataset.from_list(formatted)

def tokenize_function(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

print("Tokenizing...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

split = tokenized_dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
val_dataset = split["test"]
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# === TRAINING ARGS (~4 saat) ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_steps=50,
    logging_steps=25,
    save_steps=200,
    eval_strategy="steps",
    eval_steps=200,
    save_total_limit=3,
    fp16=True,
    report_to="none",
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    max_grad_norm=1.0,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
)

total_steps = (len(train_dataset) * 5) // (2 * 8)
print("\n" + "="*50)
print("FINE-TUNING BASLADI (~4 saat)")
print(f"Toplam step: ~{total_steps}")
print(f"LoRA rank: 16")
print(f"4-bit quantization: aktif")
print(f"Gradient checkpointing: aktif")
print(f"Epochs: 5")
print("="*50 + "\n")

start = time.time()
trainer.train()
elapsed = time.time() - start
print(f"\nToplam süre: {elapsed/3600:.1f} saat")

print("\nModel kaydediliyor...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model kaydedildi: {OUTPUT_DIR}")
print("Fine-tuning tamamlandı!")