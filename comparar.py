import os

os.environ["HF_HOME"] = r"E:\workspace\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"E:\workspace\hf_cache"
os.environ["HF_DATASETS_CACHE"] = r"E:\workspace\hf_cache"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def generar_respuesta(model_path, prompt, tokenizer_base):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


prompts = [
    "Articulo 1 del Codigo Penal establece que",
    "En derecho penal chileno, el principio de legalidad significa",
    "El homicidio simple esta definido en el articulo",
    "Para que exista dolo se requiere",
    "La diferencia entre hurto y robo radica en",
]

resultados = []
resultados.append("=" * 60)
resultados.append("COMPARACION: MODELO BASE vs MODELO ENTRENADO")
resultados.append("=" * 60)

for prompt in prompts:
    resultados.append(f"\n{'='*60}")
    resultados.append(f"PROMPT: {prompt}")
    resultados.append(f"{'='*60}")

    resultados.append("\n[MODELO BASE]:")
    respuesta_base = generar_respuesta(r"E:\workspace\modelo_base", prompt, r"E:\workspace\modelo_base")
    resultados.append(respuesta_base)

    resultados.append("\n[MODELO ENTRENADO (ADAM)]:")
    respuesta_entrenado = generar_respuesta(r"E:\workspace\modelo_entrenado_adam_merged", prompt, r"E:\workspace\modelo_base")
    resultados.append(respuesta_entrenado)

output = "\n".join(resultados)
print(output)

with open(r"E:\workspace\comparacion_adam.txt", "w", encoding="utf-8") as f:
    f.write(output)

print("\n\nComparacion guardada en: E:\\workspace\\comparacion_adam.txt")
