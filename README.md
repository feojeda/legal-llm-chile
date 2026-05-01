# LLM Legal Chileno - PoC

Este proyecto es una **prueba de concepto** para crear un asistente legal especializado en derecho chileno. Se entrenan modelos de lenguaje pequenos (**Qwen 3.5 0.8B** y **Gemma 4 E2B**) mediante continual pre-training usando exclusivamente el **Codigo Penal de Chile** como corpus.

---

## Estructura del proyecto

```
.
├── codigo_penal.txt                   # Corpus limpio (~68K palabras)
├── constitucion_chile.txt             # Constitucion Politica de Chile (~56K palabras)
├── codigo_civil_chile.txt             # Codigo Civil (~148K palabras)
├── procedimiento_penal.txt            # Codigo Procesal Penal (~69K palabras)
├── procedimiento_penal.xml            # Fuente XML oficial (BCN Ley Chile)
├── corpus_legal_chileno.txt           # Corpus combinado (~341K palabras)
├── parse_bcn_xml.py                   # Parser de XML oficial a texto plano
├── comparacion.txt                    # Comparacion Qwen base vs entrenado (QLoRA)
├── comparacion_adam.txt               # Comparacion Qwen base vs entrenado (Adam)
├── comparacion_gemma4.txt             # Comparacion Gemma 4 base vs entrenado (1 ep, CP)
├── comparacion_gemma4_exp1.txt        # Comparacion Gemma 4 exp1 (1 ep, corpus expandido)
├── comparacion_gemma4_exp5.txt        # Comparacion Gemma 4 exp5 (5 ep, corpus expandido)
├── observaciones.txt                  # Analisis cualitativo (QLoRA)
├── observaciones_adam.txt             # Analisis cualitativo (Adam)
├── observaciones_gemma4.txt           # Analisis cualitativo (Gemma 4, CP)
├── observaciones_gemma4_exp1.txt      # Analisis cualitativo (Gemma 4 exp1)
├── observaciones_gemma4_exp5.txt      # Analisis cualitativo (Gemma 4 exp5)
├── entrenar_lora.py                   # Script de entrenamiento Qwen (QLoRA)
├── entrenar_adam.py                   # Script de entrenamiento Qwen (QLoRA + AdamW8bit)
├── entrenar_gemma4.py                 # Script de entrenamiento Gemma 4 (QLoRA)
├── comparar.py                        # Script de comparacion Qwen
├── comparar_gemma4.py                 # Script de comparacion Gemma 4 (CP)
├── comparar_gemma4_exp1.py            # Script de comparacion Gemma 4 exp1
├── comparar_gemma4_exp5.py            # Script de comparacion Gemma 4 exp5
├── descargar_modelo_base.py           # Script para descargar modelo base Qwen
├── descargar_gemma4.py                # Script para descargar modelo base Gemma 4
├── extract_pdf.py                     # Script para extraer texto del PDF
├── limpiar_texto.py                   # Script para limpiar el corpus
├── modelo_base/                       # Modelo original Qwen3.5-0.8B
├── modelo_entrenado/                  # Adaptadores LoRA (PEFT)
├── modelo_entrenado_merged/           # Modelo completo mergeado
├── modelo_entrenado_adam/             # Adaptadores LoRA entrenados con Adam
├── modelo_entrenado_adam_merged/      # Modelo mergeado con Adam
├── gemma4_base/                       # Modelo original Gemma 4 E2B
├── modelo_gemma4_entrenado/           # Adaptadores LoRA Gemma 4 (CP, 1 ep)
├── modelo_gemma4_merged/              # Modelo mergeado Gemma 4 (incompleto, usar base+adapter)
├── modelo_gemma4_exp1_entrenado/      # Adaptadores LoRA Gemma 4 (corpus expandido, 1 ep)
├── modelo_gemma4_exp1_merged/         # Modelo mergeado Gemma 4 exp1
├── modelo_gemma4_exp5_entrenado/      # Adaptadores LoRA Gemma 4 (corpus expandido, 5 ep)
├── modelo_gemma4_exp5_merged/         # Modelo mergeado Gemma 4 exp5
├── resultados_lora/                   # Checkpoints del entrenamiento
├── resultados_adam/                   # Checkpoints del entrenamiento con Adam
├── resultados_gemma4/                 # Checkpoints del entrenamiento Gemma 4 (CP)
├── resultados_gemma4_exp1/            # Checkpoints Gemma 4 exp1
├── resultados_gemma4_exp5/            # Checkpoints Gemma 4 exp5
└── web/                               # App web comparador interactivo
    ├── app.py                         # Backend FastAPI
    └── static/
        └── index.html                 # Frontend
```

---

## Requisitos

- **Hardware:** GPU con al menos 8GB VRAM (Qwen) o 12GB VRAM (Gemma 4 en 4-bit)
- **Software:** Python 3.10+, PyTorch, Transformers, PEFT, bitsandbytes
- **Disco:** ~15GB libres (modelos base + checkpoints + cache)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install transformers datasets accelerate peft bitsandbytes
```

---

## Como ejecutar

### 1. Descargar modelo base
```bash
python descargar_modelo_base.py
```

### 2. Extraer y limpiar corpus (si tienes PDF)
```bash
python extract_pdf.py
python limpiar_texto.py
```

### 3. Entrenar Qwen con QLoRA
```bash
python entrenar_lora.py
```

### 4. Entrenar Gemma 4 con QLoRA
```bash
python entrenar_gemma4.py
```

### 5. Comparar resultados
```bash
python comparar.py
python comparar_gemma4.py
```

---

## Web App - Comparador interactivo

Se incluye una aplicacion web para comparar modelos en tiempo real desde el navegador.

### Requisitos
```bash
pip install fastapi uvicorn python-multipart
```

### Ejecutar
```bash
cd web
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Usar
1. Abre http://localhost:8000 en tu navegador
2. Escribe un prompt legal en el textarea
3. Selecciona los modelos a comparar (Base, QLoRA, Adam, Gemma 4 Base, Gemma 4 Entrenado)
4. Ajusta parametros si deseas (temperature, max tokens)
5. Click en **Generar comparacion**

La primera vez que uses un modelo, tardara ~1-2 minutos en cargar a VRAM. Las siguientes comparaciones seran instantaneas.

**Nota:** Seleccionar multiples modelos simultaneamente consume mas VRAM. Con una RTX 3060 12GB puedes cargar los modelos Qwen (~6-7GB total) o Gemma 4 (~8-10GB) sin problemas, pero evita cargar todos a la vez.

---

## Resultados de la comparacion

| Aspecto | Modelo Base | Modelo Entrenado |
|---|---|---|
| **Terminos legales chilenos** | No. Menciona leyes de Espana/Argentina | Intenta usar vocabulario penal chileno |
| **Cita de articulos** | Inventa leyes extranjeras (Ley 37336) | Menciona "Art. 120" (cercano pero no exacto) |
| **Estilo formal/juridico** | Conversacional, con razonamiento | Repetitivo, imita estructura de articulos |
| **Espanol coloquial** | Correcto | Muestra olvido catastrofico (mezcla portugues) |
| **Overfitting** | No aplica | Severo: repite frases verbatim |
| **Alucinaciones** | Generalistas | Dentro del dominio legal pero inventadas |

---

## Hallazgos clave

1. **El modelo adopta vocabulario legal chileno.** Responde con terminos como "delitos de orden publico", "prision", "Codigo Penal".

2. **Memorizacion parcial con overfitting severo.** En prompts sobre "Articulo 1" o "homicidio", repite la misma frase 10+ veces. Esto es esperado con un corpus pequeno.

3. **Olvido catastrofico del espanol.** En el prompt sobre "dolo", el modelo entrenado mezcla **portugues** ("tenham o dolo", "de um crime"). El modelo perdio parte de su conocimiento previo del espanol general.

4. **Alucinaciones legales persistentes.** Ningun modelo genera respuestas 100% correctas. El entrenado inventa "penas de 50 anos" o codigos de barras llamados "Hurtos y Rotos".

---

## Futuras mejoras: Uso de Adam (AdamW) para optimizacion

En este experimento usamos el **optimizador por defecto de `Trainer`** (AdamW con configuracion estandar). Para futuras iteraciones, se pueden explorar las siguientes mejoras con Adam:

### 1. AdamW con weight decay diferenciado
En lugar de aplicar weight decay uniforme, se puede usar **decoupled weight decay** con valores mayores en las capas de atencion que en las capas de embedding, protegiendo mejor el conocimiento previo del modelo:

```python
from transformers import Trainer, TrainingArguments

# Separar parametros para weight decay diferenciado
decay_params = [p for n, p in model.named_parameters() if "bias" not in n and "norm" not in n]
nodecay_params = [p for n, p in model.named_parameters() if "bias" in n or "norm" in n]

optimizer_grouped_parameters = [
    {"params": decay_params, "weight_decay": 0.1},
    {"params": nodecay_params, "weight_decay": 0.0},
]

from torch.optim import AdamW
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-4, betas=(0.9, 0.999), eps=1e-8)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset,
    data_collator=data_collator,
    optimizers=(optimizer, None),  # None = scheduler por defecto
)
```

### 2. Adam 8-bit (AdamW8bit) con bitsandbytes
Para reducir drasticamente el consumo de VRAM durante el entrenamiento full (sin QLoRA), se puede usar **Adam 8-bit**:

```python
import bitsandbytes as bnb

optimizer = bnb.optim.AdamW8bit(
    model.parameters(),
    lr=2e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.01,
    optim_bits=8,
)
```

Esto permite hacer **full fine-tuning** de modelos de ~1B parametros en GPUs de 8-12GB.

### 3. AdamW con schedule de learning rate coseno + warmup largo
Para continual pre-training, un warmup mas largo (10-20% de los steps totales) ayuda a preservar conocimiento previo:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
# Warmup manual o via get_cosine_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,   # 10-20% del total
    num_training_steps=5000,
)
```

### 4. Lion Optimizer (alternativa a Adam)
Si bien no es Adam, **Lion** (Evolved Sign Momentum) ha mostrado mejores resultados en LLMs con menos memoria:

```bash
pip install lion-pytorch
```

```python
from lion_pytorch import Lion

optimizer = Lion(
    model.parameters(),
    lr=3e-7,      # Lion requiere LR ~10x menor que Adam
    weight_decay=0.1
)
```

### Recomendacion final
Para esta PoC, **QLoRA + AdamW8bit** seria el siguiente paso si se quiere pasar de adaptadores a full fine-tuning sin aumentar la VRAM. Si se mantiene QLoRA, usar **AdamW con weight decay diferenciado** y **warmup del 20%** mejoraria la estabilidad y reduciria el olvido catastrofico.

---

## Experimento adicional: QLoRA + AdamW8bit

Se ejecuto una segunda ronda de entrenamiento aplicando las mejoras de Adam documentadas arriba:

- **Optimizador:** `bnb.optim.AdamW8bit` con weight decay diferenciado
- **Scheduler:** Coseno con warmup del 10%
- **Epochs:** 5 (vs 3 en la primera ronda)
- **LR:** 1e-4 (vs 2e-4 anterior)
- **Loss final:** ~1.57 (vs ~1.87 en QLoRA estandar)

### Hallazgos del experimento Adam

| Aspecto | QLoRA estandar | QLoRA + AdamW8bit |
|---|---|---|
| **Loss final** | ~1.87 | ~1.57 |
| **Contaminacion idiomatica** | Portugues en prompts de dolo | Menos contaminacion |
| **Estilo juridico** | Basico | Mas formal, estructura de articulos |
| **Alucinaciones** | Severas | Persisten, pero mas cercanas al dominio |
| **Overfitting** | Severo | Severo (inevitable con corpus pequeno) |

**Conclusion:** AdamW8bit mejor marginalmente la calidad del entrenamiento, pero el cuello de botella principal sigue siendo el **tamano del corpus**. Requiere corpus 10x-100x mayor para eliminar overfitting y alucinaciones.

### Como ejecutar el experimento Adam
```bash
python entrenar_adam.py
```

---

## Experimento adicional: Gemma 4 E2B + QLoRA

Se entreno **Gemma 4 E2B** (modelo multimodal de ~5B parametros, de los cuales ~4.6B son de texto) con el mismo corpus del Codigo Penal:

- **Optimizador:** AdamW (por defecto de Trainer)
- **Precision:** bf16 activado (soportado por RTX 3060 Ampere)
- **Epochs:** 1 (125 steps, ~16 minutos)
- **Batch:** 4 con acumulacion 2 (batch efectivo 8)
- **LoRA:** r=64, alpha=16, target modules estandar
- **Loss final:** ~2.01

### Hallazgos del experimento Gemma 4

| Aspecto | Gemma 4 Base | Gemma 4 Entrenado |
|---|---|---|
| **Estilo** | Conversacional/generalista | Imita estructura de articulos del Codigo Penal |
| **Citas legales** | Inventa leyes extranjeras | Inventa leyes chilenas con numeros plausibles |
| **Overfitting** | No aplica | Severo: repite frases verbatim del corpus |
| **Olvido catastrofico** | No aplica | No mezcla otros idiomas, pero pierde fluidez general |
| **Alucinaciones** | Generalistas | Dentro del dominio legal chileno |

### Problemas tecnicos resueltos

1. **PEFT no soporta `Gemma4ClippableLinear`:** se parcheo manualmente reemplazandolos por `Linear4bit` antes de aplicar LoRA.
2. **`prepare_model_for_kbit_training` causa OOM:** convierte `embed_tokens_per_layer` (2.35B params) a fp32, duplicando VRAM. Se omitio y el entrenamiento fue estable.
3. **Modelo mergeado incompleto:** `merge_and_unload()` no incluye las torres multimodales ni el `lm_head` compartido, por lo que para inferencia se recomienda cargar **base + adapter** con `PeftModel.from_pretrained`.

### Como ejecutar el experimento Gemma 4
```bash
python entrenar_gemma4.py
python comparar_gemma4.py
```

---

## Experimento: Corpus expandido (4 textos legales)

Para superar la limitacion del corpus unico (Codigo Penal), se expandio el dataset con 3 textos adicionales obtenidos de fuentes oficiales:

- **Constitucion Politica de la Republica de Chile** (~56K palabras) — descargada de Wikisource
- **Codigo Civil** (~148K palabras) — descargado de lexoffice.cl
- **Codigo Procesal Penal** (~69K palabras) — obtenido via API oficial del BCN (`leychile.cl`)

Total del corpus expandido: **~341K palabras** (~2.2 MB).

### Proceso de obtencion del Codigo Procesal Penal
```bash
# Descargar XML oficial desde BCN Ley Chile
python -c "import urllib.request; urllib.request.urlretrieve('http://www.leychile.cl/Consulta/obtxml?opt=7&idNorma=176595', 'procedimiento_penal.xml')"

# Parsear a texto plano
python parse_bcn_xml.py
```

El parser `parse_bcn_xml.py` extrae recursivamente los nodos `<Texto>` de la estructura XML oficial y genera `procedimiento_penal.txt` limpio.

### Combinacion del corpus
```bash
python -c "
files = [
    ('CONSTITUCION POLITICA DE LA REPUBLICA DE CHILE', 'constitucion_chile.txt'),
    ('CODIGO CIVIL', 'codigo_civil_chile.txt'),
    ('CODIGO PENAL', 'codigo_penal.txt'),
    ('CODIGO PROCESAL PENAL', 'procedimiento_penal.txt'),
]
with open('corpus_legal_chileno.txt', 'w', encoding='utf-8') as out:
    for title, path in files:
        out.write('\n' + '='*60 + '\n')
        out.write(title + '\n')
        out.write('='*60 + '\n\n')
        with open(path, 'r', encoding='utf-8') as f:
            out.write(f.read())
        out.write('\n')
"
```

---

## Experimento adicional: Gemma 4 + Corpus expandido

Se entreno Gemma 4 E2B con el corpus expandido en dos configuraciones para evaluar el impacto del numero de epocas.

### Exp1 — 1 epoca (verificacion rapida)
- **Pasos:** 520
- **Tiempo:** ~68 minutos
- **Loss final:** 2.456
- **Script:** `entrenar_gemma4.py` (con `num_train_epochs=1` apuntando a `corpus_legal_chileno.txt`)

### Exp5 — 5 epocas
- **Pasos:** 2.600 (520 x 5)
- **Tiempo:** ~5 horas 7 minutos
- **Loss final:** 1.788 (epoca 1 ~2.05, epoca 5 ~1.26)
- **Script:** `entrenar_gemma4.py` (con `num_train_epochs=5`)

### Hallazgos del corpus expandido

| Aspecto | Gemma 4 Base | Exp1 (1 ep) | Exp5 (5 ep) |
|---|---|---|---|
| **Dominio multilegal** | Solo generalidades | Responde sobre CP, CC, CPP y Constitucion | Responde pero con memorizacion cruzada |
| **Principio de legalidad** | Confuso | Bueno | **EXCELENTE** (casi literal del CP) |
| **Hurto vs robo** | Correcto pero mezclado | Correcto en idea, articulos dudosos | **EXCELENTE** (precision conceptual) |
| **Homicidio simple** | Alucinacion (Art. 301/313) | Coherente pero numero dudoso | **Alucinacion severa** (Art. 1321 = incendio) |
| **Matrimonio (Art. 144 CC)** | Moderno/generico | Cercano al texto legal | **Arcaico** (texto del siglo XIX) |
| **Constitucion (legislar)** | Menciona camaras | **EXCELENTE** (Congreso Nacional bicameral) | Deriva a sufragio/electores |
| **Overfitting** | No aplica | Moderado | **Severo** (bloques verbatim descontextualizados) |
| **Olvido catastrofico** | No aplica | No observado | No observado |

### Conclusion del experimento expandido

- **1 epoca es el punto optimo** para este tamano de corpus (~341K palabras). El modelo adopta el vocabulario y la estructura de los 4 codigos sin caer en sobreajuste severo.
- **5 epocas mejoran la memorizacion literal** en algunos temas (legalidad, hurto/robo) pero generan **alucinaciones cruzadas** (mezclar incendio con homicidio, definiciones antiguas con modernas) por sobreajuste.
- El corpus expandido demuestra valor agregado claro: el modelo ahora razona sobre Constitucion, Civil, Penal y Procesal Penal.

### Como ejecutar
```bash
# 1 epoca (verificacion)
python entrenar_gemma4.py
python comparar_gemma4_exp1.py

# 5 epocas (ajustar num_train_epochs=5 en el script)
python entrenar_gemma4.py
python comparar_gemma4_exp5.py
```

---

## Advertencia

> **Este es un experimento educativo.** El modelo resultante **NO debe usarse para asesoria legal real** sin supervision humana. Para un producto serio se necesitaria:
> - Corpus 10x-100x mayor (incluir Codigo Civil, fallos, doctrina)
> - Instruction tuning + RAG (Retrieval-Augmented Generation)
> - Verificacion humana obligatoria de todas las respuestas

---

## Licencia

MIT - Uso educativo y de investigacion.
