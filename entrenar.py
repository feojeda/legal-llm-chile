import os
import shutil

# Forzar cache en disco E:
os.environ["HF_HOME"] = r"E:\workspace\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"E:\workspace\hf_cache"
os.environ["HF_DATASETS_CACHE"] = r"E:\workspace\hf_cache"

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch


def main():
    print("Cargando modelo base...")
    model_path = r"E:\workspace\modelo_base"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    )
    model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Cargar corpus manualmente
    corpus_path = r"E:\workspace\codigo_penal.txt"
    print(f"Cargando corpus desde: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Tokenizando corpus completo...")
    tokenized = tokenizer(text, add_special_tokens=False, truncation=False)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    max_length = 128

    # Dividir en bloques contiguos
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

    # Configurar entrenamiento
    output_dir = r"E:\workspace\resultados_entrenamiento"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_steps=10,
        logging_steps=1,
        save_steps=100,
        save_total_limit=2,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
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
    )

    print("Iniciando entrenamiento...")
    trainer.train()

    # Guardar modelo entrenado
    modelo_entrenado_dir = r"E:\workspace\modelo_entrenado"
    if os.path.exists(modelo_entrenado_dir):
        shutil.rmtree(modelo_entrenado_dir)
    model.save_pretrained(modelo_entrenado_dir)
    tokenizer.save_pretrained(modelo_entrenado_dir)

    print(f"Modelo entrenado guardado en: {modelo_entrenado_dir}")
    print("Entrenamiento completado.")


if __name__ == "__main__":
    main()
