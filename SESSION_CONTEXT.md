# CONTEXT SESSION - Legal LLM Chile Project
**Date saved:** 2026-05-01  
**Session type:** OpenCode CLI conversation with user feojeda  
**Purpose:** Preserve full project state and conversation context for resumption

---

## 1. PROJECT OVERVIEW

Training small LLMs (Qwen 3.5 0.8B and Gemma 4 E2B) on Chilean legal corpus via QLoRA continual pre-training, with web UI for side-by-side inference.

**Repository:** https://github.com/feojeda/legal-llm-chile  
**Working directory:** `E:\workspace`  
**GPU:** RTX 3060 12GB  
**OS:** Windows, Python 3.14.4, CUDA 13.1

---

## 2. HARDWARE & ENVIRONMENT CONSTRAINTS

- **GPU:** RTX 3060 12GB VRAM
- **Disk E:\** has ~540GB free (all models/files stored here)
- **Disk C:\** is full - never install anything there
- **HF cache:** `E:\workspace\hf_cache`
- Use QLoRA (4-bit) only due to VRAM constraints
- Do NOT use `prepare_model_for_kbit_training()` with Gemma 4 (causes OOM by converting embed_tokens_per_layer to fp32)
- Gemma 4 base 4-bit uses ~7.3 GB VRAM; backward with LoRA + bs=1 uses ~10 GB

---

## 3. CORPUS FILES

| File | Words | Source | Status |
|---|---|---|---|
| `codigo_penal.txt` | ~68K | PDF extraction | Original corpus |
| `constitucion_chile.txt` | ~56K | Wikisource | Downloaded and cleaned |
| `codigo_civil_chile.txt` | ~148K | lexoffice.cl | Downloaded and cleaned |
| `procedimiento_penal.txt` | ~69K | BCN API (XML) | Parsed from `procedimiento_penal.xml` |
| `corpus_legal_chileno.txt` | ~341K | Combined | **Main training corpus** |

**BCN API endpoint:** `http://www.leychile.cl/Consulta/obtxml?opt=7&idNorma=<ID>`  
**Example:** idNorma=176595 for Ley 19.696 (Codigo Procesal Penal)

**Parser script:** `parse_bcn_xml.py` - extracts `<Texto>` nodes recursively from BCN XML schema (`http://www.leychile.cl/esquemas`)

---

## 4. MODELS & TRAINING EXPERIMENTS

### Qwen 3.5 0.8B Experiments

| Experiment | Script | Output Dir | Epochs | Loss | Notes |
|---|---|---|---|---|---|
| QLoRA baseline | `entrenar_lora.py` | `modelo_entrenado/` / `modelo_entrenado_merged/` | 3 | ~1.87 | First experiment |
| QLoRA + AdamW8bit | `entrenar_adam.py` | `modelo_entrenado_adam/` / `modelo_entrenado_adam_merged/` | 5 | ~1.57 | Lower LR (1e-4), cosine+warmup |

### Gemma 4 E2B Experiments

| Experiment | Script | Output Dir | Epochs | Loss | Corpus | Notes |
|---|---|---|---|---|---|---|
| Gemma4 CP only | `entrenar_gemma4.py` (original) | `modelo_gemma4_entrenado/` | 1 | ~2.01 | CP only | First Gemma experiment |
| **Gemma4 Exp1** | `entrenar_gemma4.py` | `modelo_gemma4_exp1_entrenado/` / `modelo_gemma4_exp1_merged/` | 1 | 2.456 | Expanded | **BEST MODEL** |
| Gemma4 Exp5 | `entrenar_gemma4.py` | `modelo_gemma4_exp5_entrenado/` / `modelo_gemma4_exp5_merged/` | 5 | 1.788 | Expanded | Overfitting issues |

### Key Training Parameters (Gemma 4)
- **Model:** `google/gemma-4-E2B-it` (~5B params, ~4.6B text)
- **Quantization:** 4-bit NF4, bfloat16 compute, double quant
- **LoRA:** r=64, alpha=16, dropout=0.05
- **Target modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Trainable params:** 119,439,360 (~2.29% of total)
- **Batch:** 4 per device, grad accumulation 2 (effective 8)
- **LR:** 2e-4
- **Max sequence length:** 128
- **Key workaround:** MUST patch `Gemma4ClippableLinear` → `Linear4bit` before `get_peft_model()`
- **Key omission:** MUST skip `prepare_model_for_kbit_training()` to avoid OOM

---

## 5. CRITICAL TECHNICAL FIXES & WORKAROUNDS

### Gemma 4 + PEFT Compatibility
```python
# MUST run this BEFORE get_peft_model()
from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

def patch_clippable(module):
    for n, child in list(module.named_children()):
        if isinstance(child, Gemma4ClippableLinear):
            setattr(module, n, child.linear)
        else:
            patch_clippable(child)

patch_clippable(model)
```

### Gemma 4 Inference
- **DO NOT use merged model** for inference (drops vision/audio towers and shared lm_head)
- **USE:** Base model + adapter via `PeftModel.from_pretrained(model_base, adapter_path)`
- See `comparar_gemma4.py`, `comparar_gemma4_exp1.py`, `comparar_gemma4_exp5.py` for working examples

### Qwen 3.5 Fast-Path Issue
- Qwen 3.5 has fast-path attention that crashes with fp16/bf16 `clip_grad_norm_`
- Solution: Use QLoRA (4-bit) instead of full fine-tuning

---

## 6. COMPARISON & OBSERVATION FILES

| File | Description |
|---|---|
| `comparacion.txt` | Qwen base vs QLoRA trained |
| `comparacion_adam.txt` | Qwen base vs AdamW8bit trained |
| `comparacion_gemma4.txt` | Gemma4 base vs CP-trained (1 ep) |
| `comparacion_gemma4_exp1.txt` | Gemma4 base vs Exp1 (1 ep, expanded) |
| `comparacion_gemma4_exp5.txt` | Gemma4 base vs Exp5 (5 ep, expanded) |
| `observaciones.txt` | QLoRA qualitative analysis |
| `observaciones_adam.txt` | AdamW8bit qualitative analysis |
| `observaciones_gemma4.txt` | Gemma4 CP qualitative analysis |
| `observaciones_gemma4_exp1.txt` | Exp1 qualitative analysis |
| `observaciones_gemma4_exp5.txt` | Exp5 qualitative analysis |

### Key Findings Summary
- **1 epoch is optimal** for ~341K word corpus. Best balance of domain learning and generalization.
- **5 epochs causes overfitting:** verbatim memorization, cross-source hallucinations (incendio↔homicidio), archaic text reproduction.
- Expanded corpus (4 legal texts) drastically improves multi-domain legal knowledge.
- Exp1 handles Constitution, Civil Code, Penal Code, and Procedural Penal Code coherently.

---

## 7. WEB APPLICATION

**Location:** `E:\workspace\web\`  
**Backend:** `E:\workspace\web\app.py` (FastAPI)  
**Frontend:** `E:\workspace\web\static\index.html`

### Current Status
- **NOT RUNNING** as of end of session
- Supports 5 models: Qwen Base, Qwen QLoRA, Qwen Adam, Gemma4 Base, Gemma4 Trained (CP only)
- **NEEDS UPDATE** to include Gemma4 Exp1 model (the best one)

### How to Start
```powershell
cd E:\workspace\web
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Cloudflare Tunnel (for public access)
- `cloudflared.exe` available in `E:\workspace\web\`
- To expose: `cloudflared tunnel --url http://localhost:8000`

---

## 8. PENDING TASKS / TODO

All items from previous todo list were completed, but the following remain for future sessions:

### High Priority
1. **Update web app** to include Gemma4 Exp1 model option (best model)
2. **Test web app** with all 5 models after adding Exp1
3. **Upload adapters to Hugging Face** - user expressed interest:
   - Create HF account: https://huggingface.co
   - Generate token at https://huggingface.co/settings/tokens
   - Upload `modelo_gemma4_exp1_entrenado/` (not merged model)
   - Repo name suggestion: `feojeda/gemma-4-legal-chile-lora`
   - Must include Gemma license terms

### Medium Priority
4. **Fetch more BCN laws** via API for even larger corpus:
   - Labor Code (Codigo del Trabajo)
   - Commercial Code (Codigo de Comercio)
   - Organic Constitutional Code (Codigo Organico de Tribunales)
5. **Try different LoRA configs** on Exp1:
   - Increase r to 128 or 256
   - Add `lm_head` and `embed_tokens` to target modules
   - Try different alpha values
6. **Implement RAG** (Retrieval-Augmented Generation) using the corpus instead of continual pre-training

### Low Priority / Ideas
7. **Quantize merged model to GGUF** for llama.cpp/Ollama compatibility
8. **Instruction tuning** on top of Exp1 adapters with Q&A pairs from legal texts
9. **Create dataset on HF** with the cleaned legal corpus for community use

---

## 9. SCRIPTS REFERENCE

| Script | Purpose |
|---|---|
| `entrenar_lora.py` | Qwen QLoRA baseline |
| `entrenar_adam.py` | Qwen QLoRA + AdamW8bit |
| `entrenar_gemma4.py` | Gemma4 QLoRA (currently points to corpus_legal_chileno.txt and saves to exp1/exp5 dirs depending on last edit) |
| `comparar.py` | Qwen CLI comparison |
| `comparar_gemma4.py` | Gemma4 CP CLI comparison |
| `comparar_gemma4_exp1.py` | Gemma4 Exp1 CLI comparison |
| `comparar_gemma4_exp5.py` | Gemma4 Exp5 CLI comparison |
| `probar_gemma4_exp1.py` | **Interactive CLI inference with Exp1** |
| `parse_bcn_xml.py` | Parse BCN Ley Chile XML to plain text |
| `extract_pdf.py` | Extract text from PDF |
| `limpiar_texto.py` | Clean/normalize corpus text |
| `descargar_modelo_base.py` | Download Qwen base |
| `descargar_gemma4.py` | Download Gemma4 base |

---

## 10. GIT REPOSITORY STATUS

**Remote:** `https://github.com/feojeda/legal-llm-chile.git`  
**Last commit:** `280b3a6` - "Add expanded legal corpus (Constitucion, Civil, Procesal Penal) and Gemma 4 experiments"  
**Branch:** master  
**Files tracked:** Scripts, comparisons, observations, parsers, corpus text files, README, .gitignore  
**Files ignored:** All model dirs, checkpoints, cache, large binaries (see .gitignore)

---

## 11. LAST CONVERSATION TOPICS

1. User asked to test the 1-epoch model (exp1) - we created `probar_gemma4_exp1.py` and ran it successfully
2. User asked about uploading to Hugging Face - explained adapters vs full model, licensing, and upload process
3. User asked if OpenCode web is running - confirmed it is NOT running
4. User wants to preserve session context - this file is the result

---

## 12. NEXT IMMEDIATE ACTIONS (when session resumes)

The user wants to start the web app version but didn't want to lose context. When resuming:
1. Read this file first
2. Ask user if they want to:
   a) Start the web app with current models
   b) Update web app to include Gemma4 Exp1 first
   c) Upload adapters to Hugging Face first
   d) Something else

---

## 13. IMPORTANT FILE PATHS (absolute)

```
E:\workspace\corpus_legal_chileno.txt
E:\workspace\modelo_gemma4_exp1_entrenado\
E:\workspace\modelo_gemma4_exp1_merged\
E:\workspace\modelo_gemma4_exp5_entrenado\
E:\workspace\modelo_gemma4_exp5_merged\
E:\workspace\gemma4_base\
E:\workspace\web\app.py
E:\workspace\web\static\index.html
E:\workspace\hf_cache\
```

---

*End of context preservation file. Read this at the beginning of the next session to resume work.*
