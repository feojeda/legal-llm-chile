# Plan de Investigación: Continual Pre-Training de LLMs para Dominio Legal Chileno

**Proyecto:** Legal LLM Chile  
**Repo:** https://github.com/feojeda/legal-llm-chile  
**Target:** arXiv preprint (inglés principal + versión español)  
**GPU:** RTX 3060 12GB VRAM  

---

## 1. Pregunta Central de Investigación

> **¿En qué medida el continual pre-training (CPT) con legislación chilena mejora el conocimiento legal de LLMs pequeños, y cómo afectan el tamaño del modelo, la arquitectura, la cantidad de épocas y la cuantización a dicha mejora?**

---

## 2. Hipótesis Testeables

| ID | Hipótesis | Variable independiente | Variable dependiente |
|---|---|---|---|
| H1 | El CPT con corpus legal mejora el rendimiento sobre el modelo base | CPT vs no-CPT | Perplexity, ROUGE-L, exact match |
| H2 | Modelos más grandes se benefician más del CPT | Tamaño (0.8B, 4B) | Mismas métricas |
| H3 | Más épocas no siempre es mejor (sobreajuste) | Epochs (1, 3, 5) | Métricas + repetición |
| H4 | La cuantización degrada el conocimiento adquirido | 4-bit vs 16-bit inferencia | Métricas de preservación |
| H5 | Diferentes arquitecturas responden distinto al mismo CPT | Qwen vs Gemma | Métricas comparativas |

---

## 3. Corpus

| Cuerpo legal | Palabras | Fuente | Archivo |
|---|---|---|---|
| Código Penal | ~68K | PDF extraction | `codigo_penal.txt` |
| Constitución Política | ~56K | Wikisource | `constitucion_chile.txt` |
| Código Civil | ~148K | lexoffice.cl | `codigo_civil_chile.txt` |
| Código Procesal Penal | ~69K | BCN API (XML) | `procedimiento_penal.txt` |
| **Total (corpus combinado)** | **~341K** | — | `corpus_legal_chileno.txt` |

**Test set para perplexity:** último ~10% de cada cuerpo legal (~34K palabras), guardado como `eval_perplexity.txt`. No se usará en entrenamiento.

---

## 4. Diseño Experimental (Matriz)

### Modelos ya entrenados (9)

| ID | Modelo | Params | CPT | Epochs | Corpus | Archivo entrenamiento |
|---|---|---|---|---|---|---|
| M0a | Qwen 3.5 0.8B base | 0.8B | No | — | — | — |
| M0b | Gemma 4 E2B base | ~4.6B | No | — | — | — |
| M1a | Qwen 0.8B QLoRA 3ep | 0.8B | Si | 3 | CP | `entrenar_lora.py` |
| M1b | Qwen 0.8B AdamW 5ep | 0.8B | Si | 5 | CP | `entrenar_adam.py` |
| M1c | Qwen 0.8B Exp1 | 0.8B | Si | 1 | Expandido | `entrenar_qwen_exp1.py` |
| M1d | Qwen 0.8B Exp3 | 0.8B | Si | 3 | Expandido | `entrenar_qwen_exp3.py` |
| M2a | Gemma 4 CP 1ep | 4.6B | Si | 1 | CP | `entrenar_gemma4.py` |
| M2b | Gemma 4 Exp1 1ep | 4.6B | Si | 1 | Expandido | `entrenar_gemma4.py` |
| M2c | Gemma 4 Exp5 5ep | 4.6B | Si | 5 | Expandido | `entrenar_gemma4.py` |

### Modelos por entrenar (2)

| ID | Modelo | Params | CPT | Epochs | Corpus | Script |
|---|---|---|---|---|---|---|
| M3a | **Qwen 3.5 4B Exp1** | 4B | Si | 1 | Expandido | `entrenar_qwen4b.py` (nuevo) |
| M3b | **Qwen 3.5 4B Exp3** | 4B | Si | 3 | Expandido | `entrenar_qwen4b.py` (nuevo) |

### Hiperparámetros estandarizados (para comparabilidad)

| Parámetro | Valor |
|---|---|
| Quantización | 4-bit NF4, bfloat16 compute, double quant |
| LoRA r | 64 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Learning rate | 2e-4 |
| Scheduler | Cosine con warmup |
| Batch size efectivo | 8 (4 per device × 2 grad accumulation) |
| Max sequence length | 128 |
| Optimizer | AdamW8bit |

---

## 5. Métricas de Evaluación

| # | Métrica | Qué mide | Cómo se computa | Automática |
|---|---|---|---|---|
| 1 | **Perplexity** | Adaptación al dominio | Loss sobre texto legal retenido | Si |
| 2 | **MC Accuracy** | Conocimiento factual | % preguntas multiple choice correctas | Si |
| 3 | **ROUGE-L** | Similitud de contenido | ROUGE-1, ROUGE-2, ROUGE-L vs referencia | Si |
| 4 | **BERTScore** | Similitud semántica | Precision, Recall, F1 con modelo BERT | Si |
| 5 | **Exact Match** | Precisión factual | % de hechos específicos correctos | Si |
| 6 | **Repetition Rate** | Calidad de generación | N-gram diversity score | Si |
| 7 | **Hallucination Rate** | Confiabilidad | % de info fuera del corpus | Semi-auto (LLM-as-judge) |

---

## 6. Benchmark de Evaluación (100 preguntas)

### 6a. 50 Preguntas Multiple Choice

- 12-13 por cuerpo legal (CP, Constitución, CC, CPP)
- 4 opciones cada una, 1 correcta
- Extraídas directamente del corpus
- 3 niveles de dificultad: básica, intermedia, avanzada

Formato:
```json
{
  "id": "cp_001",
  "categoria": "codigo_penal",
  "pregunta": "Segun el Art. 1 del Codigo Penal, delito es...",
  "opciones": {
    "A": "texto opcion A",
    "B": "texto opcion B (correcta)",
    "C": "texto opcion C",
    "D": "texto opcion D"
  },
  "correcta": "B",
  "articulo_referencia": "Art. 1 CP",
  "dificultad": "basica"
}
```

### 6b. 50 Preguntas Abiertas con Referencia

- 12-13 por cuerpo legal
- Respuesta de referencia = texto exacto del corpus
- Tipos: definición, procedimiento, garantía, concepto

Formato:
```json
{
  "id": "cc_025",
  "categoria": "codigo_civil",
  "pregunta": "Que establece el articulo 1447 del Codigo Civil?",
  "respuesta_referencia": "Art. 1447. El contrato de compraventa...",
  "articulo_referencia": "Art. 1447 CC",
  "tipo": "definicion",
  "dificultad": "basica"
}
```

### 6c. Test Set de Perplexity

- Último ~10% de cada cuerpo legal
- ~34K palabras total
- Archivo: `eval_perplexity.txt`
- No usado en entrenamiento

---

## 7. Pipeline de Evaluación

Script `evaluar.py` que ejecuta sobre cualquier modelo y produce:

1. **Perplexity** sobre `eval_perplexity.txt`
2. **MC Accuracy** sobre las 50 preguntas multiple choice
3. **ROUGE-L / BERTScore** sobre las 50 respuestas abiertas
4. **Repetition Rate** sobre generaciones libres (10 prompts)
5. **Tabla resumen** en CSV + LaTeX

Output por modelo:
```
resultados_eval/
  M0a_qwen_base/
    perplexity.json
    mc_accuracy.json
    rouge_scores.json
    repetition.json
    resumen.csv
  M1a_qwen_lora/
    ...
```

---

## 8. Estructura del Paper (arXiv, ~10-12 páginas)

```
Title: "Continual Pre-Training of Small Language Models 
        for Chilean Legal Domain: A Systematic Evaluation"

Abstract (~200 words)

1. Introduction (1.5 pages)
   - Legal AI gap in civil law / Spanish-speaking countries
   - Research questions
   - Contributions: (1) corpus, (2) benchmark, (3) systematic evaluation

2. Related Work (1.5 pages)
   - Domain-specific LLM adaptation
   - Legal AI: ChatLaw, SaulLM, LexiLaw, etc.
   - QLoRA and parameter-efficient methods

3. Methodology (2.5 pages)
   3.1 Corpus Construction (sources, cleaning, stats)
   3.2 Models (Qwen 0.8B, Gemma 4 E2B, Qwen 4B)
   3.3 Training Protocol (QLoRA config, hyperparams)
   3.4 Evaluation Benchmark (100 questions, mixed format)
   3.5 Metrics (7 metrics defined)

4. Experiments & Results (3 pages)
   4.1 Does CPT improve legal knowledge? (H1)
   4.2 Effect of model size (H2)
   4.3 Effect of training epochs (H3)
   4.4 Cross-architecture comparison (H5)
   4.5 Quantization effects (H4)

5. Analysis (1.5 pages)
   - When CPT helps vs when it doesn't
   - Hallucination patterns
   - Repetition degradation
   - Practical implications for consumer hardware

6. Conclusions & Future Work (0.5 page)

Appendix: Full benchmark, example outputs
```

**Versión en español:** Traducción completa del paper principal.

---

## 9. Tablas de Resultados Esperadas

**Tabla 1: Perplexity** — todos los modelos, base vs trained
**Tabla 2: MC Accuracy** — global y por categoría (CP, Const, CC, CPP)
**Tabla 3: ROUGE-L / BERTScore** — respuestas abiertas
**Tabla 4: Repetition & Diversity** — generaciones libres
**Tabla 5: Resumen cruzado** — mejor modelo por métrica
**Tabla 6: H4 Cuantización** — 4-bit vs FP16 inferencia

---

## 10. Fases de Ejecución

| Fase | Tarea | Entregable | Tiempo estimado |
|---|---|---|---|
| **1a** | Crear test set perplexity | `eval_perplexity.txt` | 1 hora |
| **1b** | Crear 50 preguntas MC | `benchmark_mc.json` | 2-3 días |
| **1c** | Crear 50 preguntas abiertas | `benchmark_open.json` | 2-3 días |
| **2** | Script evaluación automática | `evaluar.py` | 1 día |
| **3** | Evaluar 9 modelos existentes | `resultados_eval/` (9 dirs) | 1 día |
| **4a** | Entrenar Qwen 4B 1 epoch | `modelo_qwen4b_exp1/` | 2-3 horas |
| **4b** | Entrenar Qwen 4B 3 epochs | `modelo_qwen4b_exp3/` | 6-9 horas |
| **4c** | Evaluar modelos nuevos | `resultados_eval/` (+2 dirs) | 0.5 día |
| **5** | Análisis + tablas LaTeX | Tablas + figuras | 1-2 días |
| **6** | Escribir paper inglés | `paper/legal_llm_chile.pdf` | 3-5 días |
| **7** | Traducir paper español | `paper/legal_llm_chile_es.pdf` | 2-3 días |

**Timeline total estimado: 3-4 semanas**

---

## 11. Ventajas Competitivas para Publicación

1. **Dominio infra-representado**: Derecho civil chileno/latinoamericano. La mayoría de legal AI papers son Common Law (EEUU/UK)
2. **Hardware de consumo**: Mostrar qué es posible con una RTX 3060 democratiza IA legal
3. **Evaluación sistemática**: Estudio controlado con variables, métricas y ablations — no solo "entrenamos y funciona"
4. **Benchmark reutilizable**: Las 100 preguntas pueden ser usadas por otros investigadores
5. **Cross-architecture**: Comparación Qwen vs Gemma bajo condiciones idénticas

---

## 12. Dependencies Necesarias

```
rouge-score          # ROUGE-L
bert-score           # BERTScore  
evaluate             # HuggingFace evaluate library
numpy                # Cálculos
```

Instalación: `pip install rouge-score bert-score evaluate`
