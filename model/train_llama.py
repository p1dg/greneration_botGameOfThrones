import torch
import glob
import pandas as pd
import numpy as np
import re
from peft import get_peft_model, PeftConfig, PeftModel, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, GenerationConfig
from trl import SFTTrainer
import datasets
from utils.warning_catcher import catch_warnings

model_name = "PY007/TinyLlama-1.1B-step-50K-105b"

@catch_warnings(log_path="./logs/warnings_train_llama_model.log")
def train_llama_model(
    dataset,
    output_dir="./models_storage",
    lora_alpha=32, 
    lora_dropout=0.05,
    lora_rank=32,
    per_device_train_batch_size=3,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    save_strategy="steps",
    save_steps=100,
    logging_steps=50,
    learning_rate=2e-3,
    max_grad_norm=0.3,
    max_steps=700,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='right')
    tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_rank,
        bias="none",
        task_type="CAUSAL_LM"
    )

    peft_model = get_peft_model(model, peft_config)

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
    )

    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=500,
        dataset_text_field='prompt',
        tokenizer=tokenizer,
        args=training_arguments
    )
    peft_model.config.use_cache = False

    trainer.train()
    
