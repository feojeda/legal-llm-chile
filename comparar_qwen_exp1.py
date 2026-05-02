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
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_base, trust_remote_code=True)
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
    # Codigo Penal
    "Articulo 1 del Codigo Penal establece que",
    "El homicidio simple esta definido en el articulo",
    # Constitucion
    "El articulo 1 de la Constitucion Politica de la Republica de Chile establece que",
    "El articulo 19 N 1 de la Constitucion garantiza",
    # Codigo Civil
    "El articulo 1447 del Codigo Civil establece que el contrato de compraventa es",
    "La posesion en derecho civil chileno se adquiere por",
    # Procedimiento Penal
    "El articulo 1 del Codigo Procesal Penal senala que",
    "En el procedimiento penal chileno, la investigacion es",
]

resultados = []
resultados.append("=" * 60)
resultados.append("COMPARACION: QWEN BASE vs QWEN EXP1 (CORPUS EXPANDIDO)")
resultados.append("=" * 60)

for prompt in prompts:
    resultados.append(f"\n{'='*60}")
    resultados.append(f"PROMPT: {prompt}")
    resultados.append(f"{'='*60}")

    resultados.append("\n[MODELO BASE QWEN]:")
    respuesta_base = generar_respuesta(r"E:\workspace\modelo_base", prompt, r"E:\workspace\modelo_base")
    resultados.append(respuesta_base)

    resultados.append("\n[QWEN EXP1 - CORPUS EXPANDIDO (1 EPOCH)]:")
    respuesta_exp1 = generar_respuesta(r"E:\workspace\modelo_qwen_exp1_merged", prompt, r"E:\workspace\modelo_base")
    resultados.append(respuesta_exp1)

output = "\n".join(resultados)

with open(r"E:\workspace\comparacion_qwen_exp1.txt", "w", encoding="utf-8") as f:
    f.write(output)

print("Comparacion guardada en: E:\\workspace\\comparacion_qwen_exp1.txt")
