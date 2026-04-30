from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

model_name = "Qwen/Qwen3.5-0.8B"
output_dir = "./modelo_base"

print(f"Descargando modelo: {model_name}")
print("Esto puede tardar varios minutos la primera vez...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Modelo base guardado en: {output_dir}")
print("Descarga completada exitosamente.")
