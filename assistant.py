import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# === CONFIG ===
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
FINETUNED_PATH = "./finetuned_model"

#=======
SYSTEM_PROMPT = """Sen uzman bir yazılım mühendisisin. Adın "Code Assistant".
Türkçe açıklama yap, kod İngilizce olsun.
SADECE TÜRKÇE cevap ver,asla başka dil kullanma.
Çalışan, test edilebilir kod ver.
Neden böyle yazdığını açıkla.
>>>>>>> b893943f309bd34c665722ceb6077ca77d160779
Best practice ve clean code prensiplerine uy."""

# === LOAD MODEL ===
print("Fine-tuned model yükleniyor...")

tokenizer = AutoTokenizer.from_pretrained(FINETUNED_PATH)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

# LoRA ağırlıklarını yükle ve GPU'ya taşı
model = PeftModel.from_pretrained(
    base_model,
    FINETUNED_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)

# LoRA katmanlarını merge et (device sorunu çözülür)
print("LoRA katmanları merge ediliyor...")
model = model.merge_and_unload()
model.eval()

print("Model hazır!")

# === CHAT ===
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

conversation_history = []

def chat(user_message, max_tokens=1024, temperature=0.3):
    conversation_history.append({"role": "user", "content": user_message})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history[-10:]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    conversation_history.append({"role": "assistant", "content": response})
    return response

# === UI ===
print("\n" + "="*60)
print("   (FINE-TUNED)")
print("  Qwen2.5-Coder-7B + LoRA Fine-tuning")
print("="*60)
print("\nKomutlar:")
print("  /quit       Cikis")
print("  /clear      Sohbet gecmisini temizle")
print("  /temp 0.5   Temperature degistir")
print("  /long       Uzun cevap (2048 token)")
print("  /short      Kisa cevap (256 token)")
print("  /paste      Cok satirli kod yapistir")
print("="*60)

max_tokens = 1024
temp = 0.3

while True:
    print()
    user_input = input("Sen > ").strip()
    
    if not user_input:
        continue
    if user_input == "/quit":
        print("Gorusuruz!")
        break
    elif user_input == "/clear":
        conversation_history.clear()
        print("  Sohbet gecmisi temizlendi.")
        continue
    elif user_input.startswith("/temp "):
        try:
            temp = float(user_input.split()[1])
            temp = max(0.1, min(2.0, temp))
            print(f"  Temperature: {temp}")
        except:
            print("  Kullanim: /temp 0.5")
        continue
    elif user_input == "/long":
        max_tokens = 2048
        print("  Max tokens: 2048")
        continue
    elif user_input == "/short":
        max_tokens = 256
        print("  Max tokens: 256")
        continue
    elif user_input == "/paste":
        print("  Kodu yapistir, bos satir ile bitir:")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        user_input = "Bu kodu incele:\n```\n" + "\n".join(lines) + "\n```"
    
    print("\nAssistant > ", end="", flush=True)
    answer = chat(user_input, max_tokens=max_tokens, temperature=temp)
    print(answer)