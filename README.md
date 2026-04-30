# LLM Legal Chileno - PoC

Este proyecto es una **prueba de concepto** para crear un asistente legal especializado en derecho chileno. Se entrena un modelo de lenguaje pequeno (**Qwen 3.5 0.8B**) mediante continual pre-training usando exclusivamente el **Codigo Penal de Chile** como corpus.

---

## Estructura del proyecto

```
.
├── codigo_penal.txt              # Corpus limpio (~68K palabras)
├── comparacion.txt               # Comparacion modelo base vs entrenado
├── observaciones.txt             # Analisis cualitativo
├── entrenar_lora.py              # Script de entrenamiento (QLoRA)
├── comparar.py                   # Script de comparacion
├── descargar_modelo_base.py      # Script para descargar modelo base
├── extract_pdf.py                # Script para extraer texto del PDF
├── limpiar_texto.py              # Script para limpiar el corpus
├── modelo_base/                  # Modelo original Qwen3.5-0.8B
├── modelo_entrenado/             # Adaptadores LoRA (PEFT)
├── modelo_entrenado_merged/      # Modelo completo mergeado
└── resultados_lora/              # Checkpoints del entrenamiento
```

---

## Requisitos

- **Hardware:** GPU con al menos 8GB VRAM (ej: RTX 3060 12GB)
- **Software:** Python 3.10+, PyTorch, Transformers, PEFT, bitsandbytes
- **Disco:** ~5GB libres (modelo base + checkpoints + cache)

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

### 3. Entrenar con QLoRA
```bash
python entrenar_lora.py
```

### 4. Comparar resultados
```bash
python comparar.py
```

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

## Advertencia

> **Este es un experimento educativo.** El modelo resultante **NO debe usarse para asesoria legal real** sin supervision humana. Para un producto serio se necesitaria:
> - Corpus 10x-100x mayor (incluir Codigo Civil, fallos, doctrina)
> - Instruction tuning + RAG (Retrieval-Augmented Generation)
> - Verificacion humana obligatoria de todas las respuestas

---

## Licencia

MIT - Uso educativo y de investigacion.
