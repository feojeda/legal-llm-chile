import os
import shutil
import math

os.environ["HF_HOME"] = r"E:\workspace\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"E:\workspace\hf_cache"
os.environ["HF_DATASETS_CACHE"] = r"E:\workspace\hf_cache"

import bitsandbytes as bnb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import torch


def main():
    print("=" * 60)
    print("ENTRENAMIENTO QWEN 0.8B - CORPUS EXPANDIDO - 3 EPOCHS")
    print("=" * 60)
    print("Modelo base: Qwen 3.5 0.8B")
    print("Corpus: corpus_legal_chileno.txt (~341K palabras)")
    print("Tecnica: QLoRA 4-bit + AdamW8bit")
    print("Epochs: 3")
    print("=" * 60)

    model_path = r"E:\workspace\modelo_base"

    # 1. Cargar modelo base en 4-bit
    print("\n[1/6] Cargando modelo base en 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Preparar para k-bit training
    print("[2/6] Preparando modelo para entrenamiento k-bit...")
    model = prepare_model_for_kbit_training(model)

    # 3. Configurar LoRA
    print("[3/6] Aplicando configuracion LoRA...")
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Cargar y tokenizar corpus expandido
    print("[4/6] Cargando corpus expandido...")
    corpus_path = r"E:\workspace\corpus_legal_chileno.txt"
    print(f"   Ruta: {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"   Longitud del texto: {len(text):,} caracteres")

    print("   Tokenizando corpus completo...")
    tokenized = tokenizer(text, add_special_tokens=False, truncation=False)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    max_length = 256
    total_length = len(input_ids)
    total_length = (total_length // max_length) * max_length

    blocks_input_ids = [
        input_ids[i : i + max_length]
        for i in range(0, total_length, max_length)
    ]
    blocks_attention_mask = [
        attention_mask[i : i + max_length]
        for i in range(0, total_length, max_length)
    ]

    data = {
        "input_ids": blocks_input_ids,
        "attention_mask": blocks_attention_mask,
    }
    lm_dataset = Dataset.from_dict(data)

    print(f"   Bloques de entrenamiento: {len(lm_dataset)}")
    print(f"   Tokens totales: {total_length:,}")

    # 5. Configurar entrenamiento
    print("[5/6] Configurando entrenamiento...")
    num_epochs = 3
    batch_size = 4
    grad_accum = 2
    total_steps = math.ceil(len(lm_dataset) / (batch_size * grad_accum)) * num_epochs
    warmup_steps = max(1, int(0.10 * total_steps))

    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation: {grad_accum}")
    print(f"   Effective batch size: {batch_size * grad_accum}")
    print(f"   Learning rate: 2e-4")
    print(f"   Total steps estimados: {total_steps}")
    print(f"   Warmup steps: {warmup_steps}")

    output_dir = r"E:\workspace\resultados_qwen_exp3"

    resume_checkpoint = None
    if os.path.exists(output_dir):
        checkpoints = sorted([d for d in os.listdir(output_dir) if d.startswith("checkpoint-")])
        if checkpoints:
            resume_checkpoint = os.path.join(output_dir, checkpoints[-1])
            print(f"   [RESUME] Reanudando desde: {resume_checkpoint}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-4,
        warmup_steps=warmup_steps,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        fp16=False,
        bf16=False,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # Optimizador AdamW8bit con weight decay diferenciado
    no_decay = ["bias", "layernorm", "norm", "embedding"]
    decay_params = []
    no_decay_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n.lower() for nd in no_decay):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": 0.01},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = bnb.optim.AdamW8bit(
        optimizer_grouped_parameters,
        lr=2e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
    )

    # 6. Entrenar
    print("\n[6/6] Iniciando entrenamiento QLoRA + AdamW8bit...")
    print("=" * 60)
    if resume_checkpoint:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()

    # Guardar resultados
    print("\n" + "=" * 60)
    print("GUARDANDO MODELOS")
    print("=" * 60)

    # Guardar solo adaptadores
    modelo_entrenado_dir = r"E:\workspace\modelo_qwen_exp3_entrenado"
    if os.path.exists(modelo_entrenado_dir):
        shutil.rmtree(modelo_entrenado_dir)
    model.save_pretrained(modelo_entrenado_dir)
    tokenizer.save_pretrained(modelo_entrenado_dir)
    print(f"[OK] Adaptadores guardados en: {modelo_entrenado_dir}")

    # Mergear y guardar modelo completo
    print("[INFO] Mergeando adaptadores con modelo base...")
    merged_model = model.merge_and_unload()
    merged_dir = r"E:\workspace\modelo_qwen_exp3_merged"
    if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"[OK] Modelo mergeado guardado en: {merged_dir}")

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"Loss final: {trainer.state.log_history[-1].get('loss', 'N/A')}")
    print(f"Adaptador:  {modelo_entrenado_dir}")
    print(f"Mergeado:   {merged_dir}")


if __name__ == "__main__":
    main()
