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
    print("Cargando modelo base en 4-bit...")
    model_path = r"E:\workspace\modelo_base"

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
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

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

    # Cargar corpus
    corpus_path = r"E:\workspace\codigo_penal.txt"
    print(f"Cargando corpus desde: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Tokenizando corpus completo...")
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

    print(f"Bloques de entrenamiento: {len(lm_dataset)}")

    # Configuracion de entrenamiento
    num_epochs = 5
    batch_size = 4
    grad_accum = 2
    total_steps = math.ceil(len(lm_dataset) / (batch_size * grad_accum)) * num_epochs
    warmup_steps = max(1, int(0.10 * total_steps))  # 10% warmup

    print(f"Steps totales estimados: {total_steps}, warmup: {warmup_steps}")

    output_dir = r"E:\workspace\resultados_adam"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=1e-4,
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

    # Weight decay diferenciado
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
        lr=1e-4,
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

    print("Iniciando entrenamiento QLoRA + AdamW8bit...")
    trainer.train()

    # Guardar adaptadores
    modelo_entrenado_dir = r"E:\workspace\modelo_entrenado_adam"
    if os.path.exists(modelo_entrenado_dir):
        shutil.rmtree(modelo_entrenado_dir)
    model.save_pretrained(modelo_entrenado_dir)
    tokenizer.save_pretrained(modelo_entrenado_dir)

    # Mergear y guardar modelo completo
    print("Mergeando adaptadores con modelo base...")
    merged_model = model.merge_and_unload()
    merged_dir = r"E:\workspace\modelo_entrenado_adam_merged"
    if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    print(f"Adaptadores guardados en: {modelo_entrenado_dir}")
    print(f"Modelo mergeado guardado en: {merged_dir}")
    print("Entrenamiento QLoRA + AdamW8bit completado.")


if __name__ == "__main__":
    main()
