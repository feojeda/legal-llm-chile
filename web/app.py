import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

os.environ["HF_HOME"] = r"E:\workspace\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"E:\workspace\hf_cache"

app = FastAPI(title="Legal LLM Chile - Comparador")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATHS = {
    "base": r"E:\workspace\modelo_base",
    "qlora": r"E:\workspace\modelo_entrenado_merged",
    "adam": r"E:\workspace\modelo_entrenado_adam_merged",
    "qwen_exp1": r"E:\workspace\modelo_qwen_exp1_merged",
    "gemma4_base": r"E:\workspace\gemma4_base",
    "gemma4_trained": r"E:\workspace\modelo_gemma4_entrenado",
    "gemma4_exp1": r"E:\workspace\modelo_gemma4_exp1_entrenado",
}

MODEL_LABELS = {
    "base": "Modelo Base (Qwen 3.5 0.8B)",
    "qlora": "Entrenado QLoRA (3 epochs, CP)",
    "adam": "Entrenado AdamW8bit (5 epochs, CP)",
    "qwen_exp1": "Qwen Exp1 (1 epoch, Corpus Expandido)",
    "gemma4_base": "Gemma 4 E2B Base",
    "gemma4_trained": "Gemma 4 Entrenado (1 epoch, CP)",
    "gemma4_exp1": "Gemma 4 Exp1 (1 epoch, Corpus Expandido)",
}

loaded_models: Dict[str, tuple] = {}


def _patch_gemma4_clippable(model):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    for n, child in list(model.named_children()):
        if isinstance(child, Gemma4ClippableLinear):
            setattr(model, n, child.linear)
        else:
            _patch_gemma4_clippable(child)


def load_model(model_key: str):
    if model_key in loaded_models:
        return loaded_models[model_key]

    path = MODEL_PATHS.get(model_key)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Modelo '{model_key}' no encontrado en {path}")

    print(f"[CARGANDO] {MODEL_LABELS[model_key]} desde {path}")

    if model_key.startswith("gemma4"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        base_path = r"E:\workspace\gemma4_base"
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
        _patch_gemma4_clippable(model)
        tokenizer = AutoTokenizer.from_pretrained(base_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if model_key in ("gemma4_trained", "gemma4_exp1"):
            model = PeftModel.from_pretrained(model, path)
            print(f"[ADAPTADOR] LoRA cargado desde {path}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    loaded_models[model_key] = (model, tokenizer)
    print(f"[LISTO] {MODEL_LABELS[model_key]} cargado en memoria")
    return model, tokenizer


class GenerateRequest(BaseModel):
    prompt: str
    models: List[str]
    max_new_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9


@app.post("/generate")
async def generate(req: GenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="El prompt no puede estar vacio")
    if not req.models:
        raise HTTPException(status_code=400, detail="Selecciona al menos un modelo")

    results = {}
    for key in req.models:
        if key not in MODEL_PATHS:
            results[key] = f"Error: modelo '{key}' no valido"
            continue

        try:
            model, tokenizer = load_model(key)
            inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remover el prompt del resultado para mostrar solo la generacion
            generated = full_text[len(req.prompt):].strip()
            if not generated:
                generated = full_text.strip()

            results[key] = {
                "label": MODEL_LABELS[key],
                "generated": generated,
                "full": full_text,
            }
        except Exception as e:
            results[key] = {
                "label": MODEL_LABELS.get(key, key),
                "generated": f"Error durante la inferencia: {str(e)}",
                "full": "",
            }

    return results


@app.get("/models")
async def list_models():
    return {
        key: {
            "label": MODEL_LABELS[key],
            "path": MODEL_PATHS[key],
            "loaded": key in loaded_models,
        }
        for key in MODEL_PATHS
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "loaded_models": list(loaded_models.keys()),
    }


STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
