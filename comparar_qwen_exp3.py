import os
import gc

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

    texto = tokenizer.decode(outputs[0], skip_special_tokens=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return texto


prompts = [
    "Articulo 1 del Codigo Penal establece que",
    "El homicidio simple esta definido en el articulo",
    "El articulo 1 de la Constitucion Politica de la Republica de Chile establece que",
    "El articulo 19 N 1 de la Constitucion garantiza",
    "El articulo 1447 del Codigo Civil establece que el contrato de compraventa es",
    "La posesion en derecho civil chileno se adquiere por",
    "El articulo 1 del Codigo Procesal Penal senala que",
    "En el procedimiento penal chileno, la investigacion es",
]

resultados = []
resultados.append("=" * 60)
resultados.append("COMPARACION: QWEN BASE vs QWEN EXP3 (3 EPOCHS)")
resultados.append("=" * 60)

for i, prompt in enumerate(prompts):
    resultados.append(f"\n{'='*60}")
    resultados.append(f"PROMPT {i+1}/8: {prompt}")
    resultados.append(f"{'='*60}")

    resultados.append("\n[MODELO BASE QWEN]:")
    respuesta_base = generar_respuesta(r"E:\workspace\modelo_base", prompt, r"E:\workspace\modelo_base")
    resultados.append(respuesta_base)

    resultados.append("\n[QWEN EXP3 - CORPUS EXPANDIDO (3 EPOCHS)]:")
    respuesta_exp3 = generar_respuesta(r"E:\workspace\modelo_qwen_exp3_merged", prompt, r"E:\workspace\modelo_base")
    resultados.append(respuesta_exp3)

    with open(r"E:\workspace\comparacion_qwen_exp3.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(resultados))
    print(f"Prompt {i+1}/8 completado - guardado parcial")

print("\nComparacion completa guardada en: E:\\workspace\\comparacion_qwen_exp3.txt")
