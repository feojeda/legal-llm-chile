import os
os.environ["HF_HOME"] = r"E:\workspace\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"E:\workspace\hf_cache"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

model_path = r"E:\workspace\modelo_base"
print("Cargando modelo...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
)
model = model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Simular un batch de entrenamiento
batch_size = 2
seq_len = 256
input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len)).to("cuda")
attention_mask = torch.ones_like(input_ids).to("cuda")

print("Calentando GPU...")
for _ in range(3):
    with torch.cuda.amp.autocast():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs.loss
    loss.backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

print("Midiendo tiempo de un step...")
torch.cuda.synchronize()
start = time.time()
with torch.cuda.amp.autocast():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
loss = outputs.loss
loss.backward()
model.zero_grad(set_to_none=True)
torch.cuda.synchronize()
end = time.time()

print(f"Tiempo por step: {end-start:.2f}s")
print(f"Loss: {loss.item():.4f}")
print(f"GPU memoria usada: {torch.cuda.memory_allocated()/1e9:.2f} GB")
