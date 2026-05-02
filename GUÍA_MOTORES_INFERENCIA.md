# Motor de Inferencia para Legal LLM Chile - Gemma 4 E2B

**Fecha:** 2026-05-02  
**Modelo base:** `google/gemma-4-E2B-it` (~5B params textuales, ~4.6B activos)  
**Adaptador:** `modelo_gemma4_exp1_entrenado/` (Exp1 - 1 epoch, corpus expandido)  
**Hardware:** RTX 3060 12GB VRAM  
**SO:** Windows 11, Python 3.14.4, CUDA 12.6

---

## 1. ESTRUCTURA DE ARCHIVOS DEL MODELO

### Modelo Base (en disco)
```
E:\workspace\gemma4_base\
├── model.safetensors          # 9.54 GB - Pesos ORIGINALES en BF16
├── config.json                # Arquitectura Gemma 4 E2B
├── tokenizer.json             # Tokenizador
├── tokenizer_config.json
├── chat_template.jinja        # Template de chat
└── generation_config.json
```
- **NO está cuantizado en disco.** Son los pesos originales descargados de Hugging Face.
- El cuantizado a 4-bit NF4 solo ocurre en **runtime** al cargar con `BitsAndBytesConfig(load_in_4bit=True)`.

### Adaptador LoRA Entrenado (Exp1 - Mejor Modelo)
```
E:\workspace\modelo_gemma4_exp1_entrenado\
├── adapter_model.safetensors  # 455.73 MB - Pesos LoRA en FP16
├── adapter_config.json        # Config LoRA (r=64, alpha=16, dropout=0.05)
├── tokenizer.json             # Tokenizador
├── tokenizer_config.json
└── chat_template.jinja
```
- **NO está cuantizado.** Es un adaptador PEFT estándar (`use_qalora: false`).
- Solo entrena ~119M parámetros (~2.29% del total).
- **Target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

### Modelo Mergeado (NO usar para inferencia)
```
E:\workspace\modelo_gemma4_exp1_merged\
```
- **PROBLEMA:** Al hacer `merge_and_unload()` con Gemma 4, se pierden las torres de visión/audio y el `lm_head` compartido.
- El modelo mergeado puede generar basura o no funcionar.
- **Siempre usar:** Base + Adapter vía `PeftModel.from_pretrained()`.

---

## 2. CONSUMO DE VRAM POR MÉTODO

| Configuración | VRAM Base Estimada | VRAM Adapter | VRAM KV Cache | Total Estimado | Status RTX 3060 12GB |
|---|---|---|---|---|---|
| **Transformers + PEFT + BnB 4-bit** | ~3.8 GB | ~0.5 GB | ~2-3 GB | **~7-9 GB** | ✅ Cómodo |
| **Transformers + PEFT + BF16 (sin cuantizar)** | ~9.5 GB | ~0.5 GB | ~3-4 GB | **~13-14 GB** | ❌ OOM |
| **vLLM + BF16** | ~9.5 GB | ~0.5 GB | ~1-2 GB* | **~11-12 GB** | ⚠️ Muy justo |
| **vLLM + AWQ/GPTQ 4-bit** | ~3.5 GB | N/A** | ~1 GB | **~5-6 GB** | ✅ Ideal |
| **Ollama + GGUF Q4_K_M** | ~3.5 GB | N/A** | ~1-2 GB | **~5-7 GB** | ✅ Ideal |
| **llama.cpp + GGUF Q4_K_M** | ~3.5 GB | N/A** | ~1-2 GB | **~5-7 GB** | ✅ Ideal |

\* vLLM gestiona KV cache de forma eficiente (PagedAttention).  
\*\* vLLM y Ollama/llama.cpp NO soportan adaptadores LoRA sobre GGUF/AWQ de forma nativa. Para usarlos necesitas **mergear el adaptador al base** primero, luego cuantizar el resultado a GGUF/AWQ/GPTQ.

---

## 3. MÉTODOS DE INFERENCIA LOCAL

### Opción A: Transformers + PEFT (Método Actual - ✅ Funciona)

Tu forma actual de cargar el modelo. Es la más compatible y la única que funciona hoy sin modificaciones mayores.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

# 1. Configurar cuantización
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# 2. Cargar modelo base
base_path = r"E:\workspace\gemma4_base"
model = AutoModelForCausalLM.from_pretrained(
    base_path,
    quantization_config=bnb_config,
    device_map="auto",
)

# 3. PATCH CRÍTICO para Gemma4 + PEFT
def patch_clippable(module):
    for n, child in list(module.named_children()):
        if isinstance(child, Gemma4ClippableLinear):
            setattr(module, n, child.linear)
        else:
            patch_clippable(child)

patch_clippable(model)

# 4. Cargar adaptador
adapter_path = r"E:\workspace\modelo_gemma4_exp1_entrenado"
model = PeftModel.from_pretrained(model, adapter_path)

# 5. Tokenizador
tokenizer = AutoTokenizer.from_pretrained(base_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 6. Inferencia
inputs = tokenizer("Artículo 1 del Código Penal establece que", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Pros:**
- ✅ Funciona hoy sin instalar nada nuevo
- ✅ Soporta adaptadores LoRA nativamente
- ✅ Cuantización BnB reduce VRAM a ~7-9 GB
- ✅ Compatible con `device_map="auto"` para múltiples GPUs

**Contras:**
- ❌ Más lento que motores especializados (vLLM, llama.cpp)
- ❌ No tiene optimizaciones como PagedAttention o batching eficiente
- ❌ BnB NF4 no es compatible con compilación torch (torch.compile)
- ❌ No soporta cuantización AWQ/GPTQ/FP8 (diferentes técnicas)

---

### Opción B: vLLM (⚠️ No funciona hoy, pero es el objetivo ideal)

**vLLM** es el motor de inferencia más popular para LLMs. Usa **PagedAttention** para manejar eficientemente la memoria KV, soporta batching continuo, y es mucho más rápido que Transformers puro.

**Problemas actuales para tu setup:**

1. **Python 3.14:** vLLM no publica wheels precompilados para Python 3.14 en Windows. `pip install vllm` descarga el código fuente y necesita compilar C++/CUDA (extremadamente difícil en Windows).
2. **Windows:** vLLM está optimizado principalmente para Linux. En Windows hay soporte pero limitado.
3. **BnB no soportado:** vLLM **NO soporta** modelos cuantizados con BitsAndBytes NF4. Soporta: FP16/BF16, AWQ, GPTQ, FP8, INT8.
4. **VRAM justo:** Con BF16 y KV cache, tu RTX 3060 12GB quedaría al límite.

**Qué necesitarías para usar vLLM:**

```bash
# 1. Crear un entorno con Python 3.11 (vLLM no soporta 3.14)
# 2. Instalar vLLM (con CUDA 12.6 wheels precompilados)
pip install vllm

# 3. NO puedes usar el adaptador directamente con vLLM.
#    Necesitas MERGEAR el adaptador al base PRIMERO, luego cuantizar a AWQ/GPTQ.
```

**Cómo servir con vLLM (teórico, cuando tengas el modelo mergeado + cuantizado):**

```bash
# Si tuvieras el modelo en BF16 (sin adapter, ya mergeado)
python -m vllm.entrypoints.openai.api_server \
    --model E:\workspace\gemma4_base \
    --dtype bfloat16 \
    --max-model-len 512 \
    --gpu-memory-utilization 0.95

# Si tuvieras el modelo cuantizado a AWQ
python -m vllm.entrypoints.openai.api_server \
    --model E:\workspace\gemma4_awq \
    --quantization awq \
    --max-model-len 1024
```

**API de vLLM (formato OpenAI):**
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4-legal",
    "prompt": "Artículo 1 del Código Penal",
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

**¿vLLM tiene web UI?**
**NO.** vLLM solo levanta una API REST compatible con OpenAI. Para UI necesitas:
- **Open-WebUI** (la más popular: `pip install open-webui` → se conecta a `http://localhost:8000/v1`)
- **Un frontend propio** (como tu `web/` actual)
- **Gradio/Streamlit**

**Pros (cuando funcione):**
- ✅ Mucho más rápido (hasta 10-20x en batch)
- ✅ PagedAttention reduce fragmentation de memoria
- ✅ Batching continuo automático
- ✅ API OpenAI estándar

**Contras:**
- ❌ No soporta BnB NF4
- ❌ No soporta adaptadores LoRA en su modo optimizado (solo en modo legacy lento)
- ❌ Requiere mergear + cuantizar a AWQ/GPTQ
- ❌ Mergear Gemma 4 tiene problemas conocidos

---

### Opción C: Ollama + GGUF (Recomendado para uso personal)

**Ollama** es la forma más fácil de correr LLMs localmente en Windows. Descarga un ejecutable, crea un `Modelfile` y listo.

**El problema:** Para usar Ollama necesitas convertir tu modelo a formato **GGUF**.

#### Paso 1: Mergear adaptador + base (cuidado con Gemma 4)

Como ya sabemos, `merge_and_unload()` con Gemma 4 tiene bugs. Necesitas mergear manualmente o usar un script que solo toque los módulos de texto:

```python
# mergear_para_gguf.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

base_path = r"E:\workspace\gemma4_base"
adapter_path = r"E:\workspace\modelo_gemma4_exp1_entrenado"
output_path = r"E:\workspace\gemma4_exp1_merged_for_gguf"

# Cargar base en FP16
model = AutoModelForCausalLM.from_pretrained(
    base_path,
    torch_dtype=torch.float16,
    device_map="cpu",  # Cargar en CPU para mergear (evita OOM)
)

# Aplicar patch
def patch_clippable(module):
    for n, child in list(module.named_children()):
        if isinstance(child, Gemma4ClippableLinear):
            setattr(module, n, child.linear)
        else:
            patch_clippable(child)

patch_clippable(model)

# Cargar adaptador
model = PeftModel.from_pretrained(model, adapter_path)

# Mergear
model = model.merge_and_unload()

# Guardar solo las capas de lenguaje (ignorar vision/audio towers si existen)
model.save_pretrained(output_path, safe_serialization=True)

# Copiar tokenizador
tokenizer = AutoTokenizer.from_pretrained(base_path)
tokenizer.save_pretrained(output_path)

print(f"Modelo mergeado guardado en: {output_path}")
```

#### Paso 2: Convertir a GGUF

Necesitas descargar `llama.cpp` y usar su script de conversión:

```bash
# Clonar llama.cpp (solo necesitas los scripts Python)
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Instalar dependencias
pip install -r requirements/requirements-convert-hf-to-gguf.txt

# Convertir a GGUF Q4_K_M (buen balance calidad/tamaño)
python convert_hf_to_gguf.py \
  E:\workspace\gemma4_exp1_merged_for_gguf \
  --outfile E:\workspace\gemma4_legal_q4_k_m.gguf \
  --outtype q4_k_m
```

#### Paso 3: Crear Modelfile y usar en Ollama

```dockerfile
# Modelfile
FROM E:\workspace\gemma4_legal_q4_k_m.gguf

TEMPLATE """{{- range .Messages }}
{{- if eq .Role "user" }}<start_of_turn>user
{{ .Content }}<end_of_turn>
<start_of_turn>model
{{- else if eq .Role "assistant" }}{{ .Content }}<end_of_turn>
{{- end }}
{{- end }}"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop <end_of_turn>
```

```bash
# Crear modelo en Ollama
ollama create legal-chile -f Modelfile

# Correr interactivo
ollama run legal-chile

# Levantar servidor API (OpenAI compatible)
ollama serve

# Probar via API
curl http://localhost:11434/api/generate -d '{
  "model": "legal-chile",
  "prompt": "Artículo 1 del Código Penal establece que",
  "stream": false
}'
```

**¿Ollama tiene web UI?**
**NO nativamente**, pero hay muchas opciones:
- **Open-WebUI**: Se conecta automáticamente a Ollama (`pip install open-webui`)
- **ChatGPT-Next-Web**: Cliente web que se conecta a APIs locales
- **Cualquier frontend OpenAI-compatible**

**Pros:**
- ✅ Muy fácil de usar (un ejecutable en Windows)
- ✅ GGUF Q4_K_M es muy eficiente (~3.5-4.5 GB VRAM)
- ✅ Soporta `llama.cpp` optimizations (flash attention, etc.)
- ✅ API simple
- ✅ No depende de Python ni PyTorch

**Contras:**
- ❌ Necesita convertir a GGUF (mergear + convertir)
- ❌ Mergear Gemma 4 tiene riesgos (puede perder calidad)
- ❌ GGUF es una cuantización de pérdida (puede degradar ligeramente vs BF16)
- ❌ No soporta adaptadores dinámicos (tienes que regenerar el GGUF para cambiar adapters)
- ❌ El template de chat de Gemma 4 puede no traducirse perfectamente a GGUF

---

### Opción D: llama.cpp Directo (Alternativa a Ollama)

Si no quieres usar Ollama (que es un wrapper de llama.cpp), puedes usar **llama.cpp** directamente.

```bash
# Descargar binario precompilado de llama.cpp para Windows
# Desde: https://github.com/ggerganov/llama.cpp/releases

# Convertir a GGUF (igual que en Opción C)
python convert_hf_to_gguf.py ...

# Correr inferencia CLI
llama-cli.exe \
  -m E:\workspace\gemma4_legal_q4_k_m.gguf \
  -p "Artículo 1 del Código Penal establece que" \
  -n 150 \
  --temp 0.7 \
  --top-p 0.9

# Levantar servidor HTTP
llama-server.exe \
  -m E:\workspace\gemma4_legal_q4_k_m.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 2048

# API: http://localhost:8080/completion
# Web UI integrada: http://localhost:8080
```

**¿llama.cpp tiene web UI?**
**SÍ.** `llama-server` incluye una web UI simple en `http://localhost:8080`.

**Pros:**
- ✅ Más control que Ollama
- ✅ Web UI integrada
- ✅ Soporta todas las optimizaciones de llama.cpp

**Contras:**
- ❌ Igual que Ollama: requiere GGUF y mergear primero
- ❌ Menos "amigable" que Ollama

---

### Opción E: Text Generation Inference (TGI) de Hugging Face

TGI es el motor de inferencia de producción de Hugging Face. Similar a vLLM pero más enfocado a despliegue en producción.

**Problemas:**
- Principalmente soportado en Linux/Docker
- No soporta BnB NF4
- No soporta adaptadores LoRA dinámicos nativamente
- Overkill para uso local personal

**NO recomendado** para tu caso de uso local.

---

### Opción F: Web App Custom (FastAPI + Gradio)

Ya tienes una web app FastAPI en `E:\workspace\web/`. Es la forma más flexible.

**FastAPI (ya lo tienes):**
- Backend en Python con Transformers + PEFT
- Carga los modelos por demanda
- API REST propia

**Gradio (alternativa frontend):**
```python
# servir_gemma4_gradio.py
import gradio as gr
from tu_script_de_carga import load_model  # Tu función actual

model, tokenizer = load_model("gemma4_exp1")

def generate(prompt, max_tokens, temperature):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(lines=3, label="Prompt legal"),
        gr.Slider(10, 512, value=150, label="Max tokens"),
        gr.Slider(0.1, 2.0, value=0.7, label="Temperature"),
    ],
    outputs=gr.Textbox(lines=10, label="Respuesta"),
    title="Legal LLM Chile - Gemma 4 Exp1",
).launch(server_name="0.0.0.0", server_port=7860)
```

```bash
pip install gradio
python servir_gemma4_gradio.py
# Abre: http://localhost:7860
```

**Pros:**
- ✅ Totalmente customizable
- ✅ Usa tu setup actual sin conversiones
- ✅ Puedes integrar múltiples modelos

**Contras:**
- ❌ No es tan eficiente como vLLM/llama.cpp
- ❌ Mantenimiento propio

---

## 4. TABLA COMPARATIVA DE MOTORES

| Motor | Velocidad | VRAM RTX 3060 | Windows | Web UI | Soporta LoRA Adapter | Requiere Conversión | Dificultad Setup |
|---|---|---|---|---|---|---|---|
| **Transformers+PEFT** | 🐢 Lenta | ✅ 7-9 GB | ✅ Sí | Manual (FastAPI) | ✅ Nativo | ❌ No | ⭐ Fácil |
| **vLLM** | 🚀 Muy rápida | ⚠️ 11-12 GB | ⚠️ Limitado | No (solo API) | ⚠️ Legacy lento | ✅ Merge+AWQ | ⭐⭐⭐ Difícil |
| **Ollama** | ⚡ Rápida | ✅ 5-7 GB | ✅ Sí | No (solo API) | ❌ No | ✅ Merge+GGUF | ⭐⭐ Medio |
| **llama.cpp** | ⚡ Rápida | ✅ 5-7 GB | ✅ Sí | ✅ Sí (server) | ❌ No | ✅ Merge+GGUF | ⭐⭐ Medio |
| **TGI** | 🚀 Muy rápida | ⚠️ 11-12 GB | ❌ No | No | ❌ No | ✅ Merge+AWQ | ⭐⭐⭐ Difícil |
| **Gradio** | 🐢 Lenta | ✅ 7-9 GB | ✅ Sí | ✅ Sí | ✅ Nativo | ❌ No | ⭐ Fácil |

---

## 5. FLUJO RECOMENDADO PARA SUBIR A HUGGING FACE

Cuando subas el modelo, **sube solo el adaptador**, no el modelo base. Esto:
1. Respeta la licencia de Google (Gemma tiene restricciones)
2. Reduce el tamaño a subir (456 MB vs 9.5 GB)
3. Permite a otros usarlo con el base oficial

### Estructura del repo en HF:
```
feojeda/gemma-4-legal-chile-lora/
├── adapter_model.safetensors     # 456 MB
├── adapter_config.json
├── tokenizer.json
├── tokenizer_config.json
├── chat_template.jinja
├── README.md                     # Instrucciones de uso
└── USAGE_EXAMPLE.py              # Código de ejemplo
```

### Cómo usar el adaptador desde HF (para otros usuarios):
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 1. Descargar base desde HF (o local)
base = "google/gemma-4-E2B-it"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb_config)

# 2. Aplicar patch de Gemma4ClippableLinear (ver arriba)
# ...

# 3. Cargar adaptador desde HF
model = PeftModel.from_pretrained(model, "feojeda/gemma-4-legal-chile-lora")
```

---

## 6. CHECKLIST DE DECISIÓN

```
┌─────────────────────────────────────────────────────────────────┐
│ ¿Quiero probar AHORA sin instalar nada nuevo?                   │
│    └─► Usa tu web app FastAPI actual (Opción A / F)             │
│                                                                 │
│ ¿Quiero el motor más rápido y estoy dispuesto a perder calidad? │
│    └─► Ruta Ollama/llama.cpp (Opción C / D)                     │
│        Necesita: mergear adapter + convertir a GGUF             │
│        Riesgo: mergear Gemma 4 puede dar problemas              │
│                                                                 │
│ ¿Quiero velocidad de producción y tengo Python 3.11?            │
│    └─► vLLM (Opción B)                                          │
│        Necesita: mergear adapter + cuantizar a AWQ/GPTQ         │
│        VRAM: muy justo en RTX 3060                              │
│                                                                 │
│ ¿Quiero máxima compatibilidad con mi setup actual?              │
│    └─► Seguir con Transformers + PEFT                           │
│        Puedes mejorar: Gradio UI, batching manual, etc.         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. NOTAS TÉCNICAS ADICIONALES

### Por qué vLLM no soporta BnB NF4
- **BnB (BitsAndBytes)** cuantiza los pesos "al vuelo" en el momento de la inferencia usando dequantización lazy. Esto es incompatible con los kernels optimizados de vLLM.
- vLLM usa cuantizaciones que requieren los pesos ya cuantizados en disco: **AWQ, GPTQ, FP8, INT8**.
- Estos formatos usan kernels CUDA específicos que vLLM implementa para máxima velocidad.

### Por qué Ollama/llama.cpp no soportan LoRA adapters dinámicos
- GGUF es un formato de inferencia estático. Los pesos están cuantizados y optimizados para lectura secuencial.
- Aunque llama.cpp tiene soporte experimental para LoRA adapters (parámetro `--lora`), es lento y poco usado.
- La forma estándar es: mergear adapter → cuantizar → GGUF.

### Diferencia entre cuantizaciones
| Técnica | Tipo | Uso | Compatibilidad |
|---|---|---|---|
| **BnB 4-bit NF4** | Runtime (carga en memoria) | Entrenamiento e inferencia con PEFT | Transformers, PEFT |
| **AWQ** | Estático (en disco) | Inferencia optimizada | vLLM, AutoAWQ |
| **GPTQ** | Estático (en disco) | Inferencia optimizada | vLLM, AutoGPTQ, transformers |
| **GGUF Q4_K_M** | Estático (en disco) | Inferencia con llama.cpp | Ollama, llama.cpp, koboldcpp |
| **FP8** | Estático (en disco) | Inferencia en GPUs modernas (Ada/Hopper) | vLLM, transformers |

---

*Documento generado el 2026-05-02. Actualizar si cambian las versiones de vLLM, Ollama o transformers.*
