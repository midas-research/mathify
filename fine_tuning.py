import pickle
import json
import json
import os
import bitsandbytes as bnb
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    AutoPeftModelForCausalLM
)
import argparse
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import TrainingArguments

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="Show Model")
parser.add_argument("--train", type=str, help="Show Dataset File")
parser.add_argument("--val", type=str, help="Show Dataset File")
parser.add_argument("--output", type=str, help="Show Fined Tuned Folder")

args = parser.parse_args()

for arg_name in vars(args):
    arg_value = getattr(args, arg_name)
    print(f'{arg_name}: {arg_value}')

MODEL_NAME = args.model
TRAIN = args.train 
VAL = args.val
OUTPUT_DIR = args.output

os.makedirs(OUTPUT_DIR, exist_ok=True)

from datasets import load_dataset
from random import randrange


def format_instruction(sample):
    prompt=f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. Beaware of wrong calculation and do not repeat it.\n\n### Instruction:\n{sample['Question']}\n\n### Input:\n{sample["Input"]}\n\n### Response: {sample["Response_accepted"]}"""
    sample["prompt"] = prompt
    return sample


def get_data():
    train_data = load_dataset("json", data_files=TRAIN)["train"]
    val_data = load_dataset("json", data_files=VAL)["train"]
    train_data = train_data.map(format_instruction)
    val_data = val_data.map(format_instruction)
    print(train_data, val_data)
    return train_data, val_data


train_data, val_data = get_data()

# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, device_map="auto", quantization_config=quant_config)
model.config.use_cache=False
model.config.pretraining_tp=1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side="right" # fix for fp16


# LoRA config based on QLoRA paper
peft_config = LoraConfig(
        lora_alpha=64,
        lora_dropout=0.1,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
)


EPOCHS = 4

MICRO_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 3e-4

args = TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=0.1,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    bf16=True,
    logging_steps=1,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=0.15,
    save_steps=0.15,
    # max_grad_norm=0.3,
    output_dir=OUTPUT_DIR,
    save_total_limit=3,
    load_best_model_at_end=True,
    logging_dir=OUTPUT_DIR,
    report_to="wandb"
)



from trl import SFTTrainer

max_seq_length = 2048 # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    dataset_text_field="prompt",
    args=args,
)

trainer.train() 

# save model
trainer.save_model(OUTPUT_DIR)

