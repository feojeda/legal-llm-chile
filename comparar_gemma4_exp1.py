import os

os.environ["HF_HOME"] = r"E:\workspace\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"E:\workspace\hf_cache"
os.environ["HF_DATASETS_CACHE"] = r"E:\workspace\hf_cache"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
import torch


def _patch_gemma4_clippable(model):
    for n, child in list(model.named_children()):
        if isinstance(child, Gemma4ClippableLinear):
            setattr(model, n, child.linear)
        else:
            _patch_gemma4_clippable(child)


def cargar_modelo_base():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        r"E:\workspace\gemma4_base",
        quantization_config=bnb_config,
        device_map="auto",
    )
    _patch_gemma4_clippable(model)
    tokenizer = AutoTokenizer.from_pretrained(r"E:\workspace\gemma4_base")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generar_respuesta(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    prompts = [
        "Articulo 1 del Codigo Penal establece que",
        "En derecho penal chileno, el principio de legalidad significa",
        "El homicidio simple esta definido en el articulo",
        "Para que exista dolo se requiere",
        "La diferencia entre hurto y robo radica en",
        "El Codigo Procesal Penal establece que la investigacion penal sera dirigida por",
        "Segun la Constitucion de Chile, la funcion de legislar corresponde a",
        "El articulo 144 del Codigo Civil define el matrimonio como",
    ]

    print("Cargando Gemma 4 base...")
    model_base, tokenizer = cargar_modelo_base()

    print("Cargando Gemma 4 exp1 entrenado (base + adaptador)...")
    model_trained = PeftModel.from_pretrained(model_base, r"E:\workspace\modelo_gemma4_exp1_entrenado")

    resultados = []
    resultados.append("=" * 60)
    resultados.append("COMPARACION: GEMMA 4 BASE vs GEMMA 4 EXP1 (1 EPOCA, CORPUS EXPANDIDO)")
    resultados.append("=" * 60)

    for prompt in prompts:
        resultados.append(f"\n{'='*60}")
        resultados.append(f"PROMPT: {prompt}")
        resultados.append(f"{'='*60}")

        resultados.append("\n[GEMMA 4 BASE]:")
        resultados.append(generar_respuesta(model_base, tokenizer, prompt))

        resultados.append("\n[GEMMA 4 EXP1 ENTRENADO]:")
        resultados.append(generar_respuesta(model_trained, tokenizer, prompt))

    output = "\n".join(resultados)
    print(output)

    with open(r"E:\workspace\comparacion_gemma4_exp1.txt", "w", encoding="utf-8") as f:
        f.write(output)

    print("\n\nComparacion guardada en: E:\\workspace\\comparacion_gemma4_exp1.txt")


if __name__ == "__main__":
    main()
