import torch
import glob
import pandas as pd
import numpy as np
import re
from peft import get_peft_model, PeftConfig, PeftModel, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, GenerationConfig
from trl import SFTTrainer
from datasets import Dataset
import warnings
import random
    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class GenerationBot:
    def __init__(
        self,
        trained_model_dir,
        role,
        model_name="PY007/TinyLlama-1.1B-step-50K-105b"
    ):
        self.config = PeftConfig.from_pretrained(trained_model_dir)
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.trained_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name_or_path,
            quantization_config=self.bnb_config,
            trust_remote_code=True,
            device_map=DEVICE
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='right')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.context_memory = ''
        self.role = role
            
    def generate_answer(self, query, max_new_tokens=250,):
        prompt = self.get_prompt(query, self.context_memory)
        encoding = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.3,
            eos_token_id=self.tokenizer.eos_token_id
        )

        outputs = self.trained_model.generate(input_ids=encoding.input_ids, generation_config=generation_config)
        text_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        ans = text_output.split(f"your role: {self.role}\n\n")[1].split("answer:")[1:]
        res = []
        for txt in ans:
            txt = txt.replace("[/INST]", " ").replace("\n", " ")
            if len(txt) > 5: 
                res.append(txt.split(".")[0])
        
        if len(res) == 1:
            answer = res[0]
            
        elif len(res) > 1:   
            answer = random.choice(res)
        else: 
            answer = "I can't undanstand you, please answer me again"
        self.context_memory = '.'.join([self.context_memory, query, answer])
        
        return answer, self.role

    def get_prompt(self, query, context,):
        prompt = f"<s>[INST]"
        prompt += f'Use the given context to guide your an about the query like indicated in your role'
        prompt += f"query: {query}\n\n"
        prompt += f"context: {context}\n\n"
        prompt += f"your role: {self.role}\n\n"
        prompt += f'answer:[/INST]</s>'

        return prompt


if __name__ == "__main__":
    pass
