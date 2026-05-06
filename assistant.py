import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# === CONFIG ===
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

SYSTEM_PROMPT = """Sen uzman bir yazılım mühendisisin. Adın "Code Assistant".

Kuralların:
- Türkçe açıklama yap, kod İngilizce olsun
- Çalışan, test edilebilir kod ver
- Neden böyle yazdığını açıkla
- Hata varsa önce hatayı açıkla, sonra düzelt
- Best practice ve clean code prensiplerine uy
- Alternatif yaklaşımlar öner
- Kısa ve öz cevaplar ver, gereksiz uzatma"""

# === LOAD ===
print("Model yükleniyor...")
print(f"({MODEL_NAME})")
print("İlk seferde ~15 GB indirecek, sabırla bekle.\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)

print(f"Model hazır!")
print(f"Parametreler: {sum(p.numel() for p in model.parameters())/1e9:.1f}B")
print(f"Cihaz: {next(model.parameters()).device}")


# === CHAT ===
conversation_history = []

def chat(user_message, max_tokens=1024, temperature=0.3):
    conversation_history.append({"role": "user", "content": user_message})
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history[-10:]
    
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
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
    
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    conversation_history.append({"role": "assistant", "content": response})
    return response


# === UI ===
print("\n" + "="*60)
print("  CODE ASSISTANT")
print("  Qwen2.5-Coder-7B | Senin kişisel kod asistanın")
print("="*60)
print("\nKomutlar:")
print("  /quit       Çıkış")
print("  /clear      Sohbet geçmişini temizle")
print("  /temp 0.5   Temperature değiştir")
print("  /long       Uzun cevap (2048 token)")
print("  /short      Kısa cevap (256 token)")
print("  /paste      Çok satırlı kod yapıştır")
print("="*60)

max_tokens = 1024
temp = 0.3

while True:
    print()
    user_input = input("Sen > ").strip()
    
    if not user_input:
        continue
    
    if user_input == "/quit":
        print("Görüşürüz!")
        break
    
    elif user_input == "/clear":
        conversation_history.clear()
        print("  Sohbet geçmişi temizlendi.")
        continue
    
    elif user_input.startswith("/temp "):
        try:
            temp = float(user_input.split()[1])
            temp = max(0.1, min(2.0, temp))
            print(f"  Temperature: {temp}")
        except:
            print("  Kullanım: /temp 0.5")
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
        print("  Kodu yapıştır, boş satır ile bitir:")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        user_input = "Bu kodu incele:\n```\n" + "\n".join(lines) + "\n```"
        print(f"  ({len(lines)} satır kod alındı)")
    
    print("\nAssistant > ", end="", flush=True)
    answer = chat(user_input, max_tokens=max_tokens, temperature=temp)
    print(answer)