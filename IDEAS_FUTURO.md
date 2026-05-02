# Ideas a Explorar

Resultados del continual pre-training con Qwen 3.5 0.8B muestran que exponer el modelo a texto legal crudo NO es suficiente para generar respuestas legalmente correctas. Se necesita un enfoque diferente.

---

## 1. Instruction Tuning con Pares Q&A

**Problema actual:** El modelo aprende patrones de tokens del corpus, pero no aprende a *responder preguntas*.

**Solucion:** Crear un dataset de pares pregunta-respuesta extraidos del corpus legal y hacer fine-tuning con formato de instruccion.

**Pasos:**
- Extraer articulos del corpus y generar pares (pregunta, respuesta) automaticamente
- Formato tipo Alpaca: `### Instruction:\n{pregunta}\n### Response:\n{respuesta}`
- Fine-tuning con QLoRA sobre el modelo base o sobre el modelo con continual pre-training
- Dataset objetivo: ~1000-5000 pares Q&A de calidad

**Modelo candidato:** Qwen 3.5 0.8B o Gemma 4 4B (mejor retencion de conocimiento)
**VRAM estimada:** ~8-10 GB (mismo QLoRA 4-bit)
**Tiempo estimado:** 1-3 horas en RTX 3060

---

## 2. RAG (Retrieval Augmented Generation)

**Problema actual:** El modelo no puede memorizar 341K palabras de texto legal con solo 0.8B parametros.

**Solucion:** No entrenar el modelo. En cambio, indexar el corpus y recuperar fragmentos relevantes en tiempo de consulta.

**Pasos:**
- Dividir corpus en chunks (500-1000 tokens)
- Generar embeddings con modelo tipo `sentence-transformers` o `BAAI/bge-m3`
- Indexar con FAISS o ChromaDB
- En consulta: recuperar top-k chunks, pasarlos como contexto al modelo
- Usar Gemma 4 Exp1 como modelo generador (mejor rendimiento)

**Ventajas:**
- Sin entrenamiento adicional
- Siempre citea fuentes reales
- Facil de actualizar (agregar nuevas leyes sin re-entrenar)
- Funciona con modelos pequenos

**Desventajas:**
- Requiere infraestructura adicional (indice, embeddings)
- Latencia mayor por retrieval

---

## 3. Modelo Mas Grande

**Problema actual:** 0.8B parametros es insuficiente para retener conocimiento legal detallado.

**Opciones:**

| Modelo | Params | VRAM QLoRA 4-bit | Notas |
|---|---|---|---|
| Qwen 3.5 1.5B | 1.5B | ~4 GB | Mejor base, mismo enfoque |
| Qwen 3.5 4B | 4B | ~6 GB | Buen balance tamano/rendimiento |
| Gemma 4 4B | 4B | ~6 GB | Ya probado, buen candidato |
| Qwen 3.5 7B | 7B | ~10 GB | Limite de RTX 3060, mejor retencion |

**Consideraciones:**
- Gemma 4 4B Exp1 ya fue el mejor modelo en pruebas anteriores
- Un modelo de 4B-7B con instruction tuning podria dar resultados significativamente mejores
- Verificar compatibilidad con transformers 5.x y bitsandbytes

---

## 4. Dataset de Evaluacion Formal

**Problema actual:** Evaluacion cualitativa (lectura subjetiva de outputs).

**Solucion:** Crear benchmark con metricas cuantitativas.

**Pasos:**
- Crear 50-100 preguntas con respuestas correctas verificables
- Metricas: exact match, ROUGE-L, BERTScore, perplexidad
- Evaluar todos los modelos con el mismo benchmark
- Comparar: base, Qwen Exp1, Qwen Exp3, Gemma4 Exp1

**Formato sugerido:**
```json
{
  "pregunta": "Que establece el Art. 1 del Codigo Penal?",
  "respuesta_correcta": "Art 1... [texto exacto]",
  "categoria": "codigo_penal",
  "tipo": "definicion"
}
```

---

## 5. Combinar Approaches: RAG + Fine-tuning

**Mejor escenario:** Fine-tuning con instruction tuning + RAG para retrieval de contexto.

1. Instruction tuning ensena al modelo a responder en formato legal
2. RAG proporciona contexto preciso y actualizado
3. El modelo sintetiza la respuesta usando el contexto proporcionado

Este es el approach que usan la mayoria de los sistemas legales AI en produccion.

---

## Prioridad Sugerida

1. **RAG** (rapido de implementar, impacto inmediato, sin entrenamiento)
2. **Instruction tuning** (requiere crear dataset, pero mejora fundamental)
3. **Modelo mas grande** (si los anteriores no son suficientes)
4. **Dataset de evaluacion** (para medir progreso objetivo)
5. **RAG + fine-tuning** (combinacion final para produccion)
