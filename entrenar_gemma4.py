import os
import shutil
import math

os.environ["HF_HOME"] = r"E:\workspace\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"E:\workspace\hf_cache"
os.environ["HF_DATASETS_CACHE"] = r"E:\workspace\hf_cache"

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import torch


def main():
    print("Cargando Gemma 4 E2B en 4-bit...")
    model_path = r"E:\workspace\gemma4_base"

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

    # Gemma 4 es multimodal (vision + audio + text).
    # Para continual pre-training de texto, aplicamos LoRA al language_model interno.
    # PEFT get_peft_model aplica al modelo completo, asi que usamos target_modules
    # con el prefijo del language_model.
    # NOTA: omitimos prepare_model_for_kbit_training porque convierte
    # embed_tokens_per_layer (2.35B params) a fp32, duplicando el uso de VRAM
    # y causando OOM en una GPU de 12 GB.

    # Gemma 4 usa Gemma4ClippableLinear en vision/audio towers; PEFT no lo soporta.
    # Reemplazamos esos wrappers por su Linear4bit interno antes de aplicar LoRA.
    print("Patching Gemma4ClippableLinear -> Linear4bit para compatibilidad con PEFT...")
    import torch.nn as nn
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

    def patch_clippable(module, parent, name):
        for n, child in list(module.named_children()):
            if isinstance(child, Gemma4ClippableLinear):
                setattr(module, n, child.linear)
            else:
                patch_clippable(child, module, n)

    patch_clippable(model, None, "")

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

    max_length = 128
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

    output_dir = r"E:\workspace\resultados_gemma4"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        fp16=False,
        bf16=True,
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

    print("Iniciando entrenamiento QLoRA Gemma 4...")
    trainer.train()

    # Guardar adaptadores
    modelo_entrenado_dir = r"E:\workspace\modelo_gemma4_entrenado"
    if os.path.exists(modelo_entrenado_dir):
        shutil.rmtree(modelo_entrenado_dir)
    model.save_pretrained(modelo_entrenado_dir)
    tokenizer.save_pretrained(modelo_entrenado_dir)

    # Mergear y guardar
    print("Mergeando adaptadores con modelo base...")
    merged_model = model.merge_and_unload()
    merged_dir = r"E:\workspace\modelo_gemma4_merged"
    if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    print(f"Adaptadores guardados en: {modelo_entrenado_dir}")
    print(f"Modelo mergeado guardado en: {merged_dir}")
    print("Entrenamiento Gemma 4 completado.")


if __name__ == "__main__":
    main()
