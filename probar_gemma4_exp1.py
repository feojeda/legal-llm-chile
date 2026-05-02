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


def cargar_modelo():
    print("Cargando Gemma 4 base + adaptadores exp1 (1 epoca, corpus expandido)...")
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

    model = PeftModel.from_pretrained(model, r"E:\workspace\modelo_gemma4_exp1_entrenado")
    print("Modelo listo. Escribe un prompt legal (o 'salir' para terminar):\n")
    return model, tokenizer


def generar(model, tokenizer, prompt, max_new_tokens=200, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    model, tokenizer = cargar_modelo()
    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo...")
            break
        if prompt.strip().lower() in ("salir", "exit", "quit"):
            break
        if not prompt.strip():
            continue
        respuesta = generar(model, tokenizer, prompt)
        print(respuesta)
        print("-" * 60)


if __name__ == "__main__":
    main()
