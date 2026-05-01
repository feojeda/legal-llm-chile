from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

os.environ["HF_HOME"] = r"E:\workspace\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"E:\workspace\hf_cache"

model_name = "google/gemma-4-E2B-it"
output_dir = r"E:\workspace\gemma4_base"

print(f"Descargando modelo: {model_name}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Modelo base guardado en: {output_dir}")
