"""
==============================================================================
Module:      transformer_pipeline.py
Description: Generative language modeling with Hugging Face Transformers:
             GPT-2 inference sampling and causal language model fine-tuning.
==============================================================================
"""

import os
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from typing import Optional


def generate_text_gpt2(prompt: str, max_length: int = 50, temperature: float = 0.7) -> str:
    """Generiert Textfortsetzungen mit dem vortrainierten GPT-2 Modell."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    inputs = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result


def finetune_gpt2_wikitext(output_dir: str = "./results", max_steps: int = 20, max_length: int = 64) -> str:
    """Demonstriert Fine-Tuning von GPT-2 auf einer Stichprobe des WikiText-2 Datensatzes."""
    print("[*] Lade GPT-2 Modell und WikiText-2 Datensatz...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # 10 % des Trainingssplits laden
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:5%]")

    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=8,
        save_strategy="no",
        logging_steps=5,
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    print(f"[*] Starte Fine-Tuning ({max_steps} Optimierungsschritte)...")
    trainer.train()

    test_prompt = "The future of artificial intelligence"
    input_ids = tokenizer.encode(test_prompt, return_tensors="pt")
    out = model.generate(input_ids, max_length=40, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)
