import json
import os
from pprint import pprint

import bitsandbytes as bnb
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset, load_from_disk
from huggingface_hub import notebook_login
from peft import (LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training)
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)

os.environ['CUDA_VISIBLE_DEVICES']='0'
model_name='tiiuae/falcon-7b'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
print('bits and bytes config loaded.')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    trust_remote_code=True,
    quantization_config=bnb_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # set padding token to end of sequence token
def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0 
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params +=param.numel()
    print(f'trainable params: {trainable_params} || all params: {all_params} || trainable: {round(100 * trainable_params /all_params, 2)} %')
model.gradient_checkpointing_enable() # trade off between using GPU so memory and efficiency 
model = prepare_model_for_kbit_training(model) # wrapper around the model - here we train in 4bit
print('model and tokenizer loaded.')

config = LoraConfig(
    r=16, # rank of the matric try to reduce it and see if results are getting better
    lora_alpha=32,
    target_modules=['query_key_value'],
    lora_dropout=.05,
    bias='none',
    task_type='CAUSAL_LM',
    base_model_name_or_path=model_name
)
print('lara config loaded.')
model = get_peft_model(model, config) # applying lora configuration on top of our model
print_trainable_parameters(model)


data = load_from_disk('data/sap_sam_2022/adrian_filter/forTraining/training')['train']
def generate_prompt(data_point):
    return f"""
<question>: {data_point['context']}
<answer>: {data_point['target']}
    """.strip()
def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenizer(full_prompt, padding=True, truncation=True)
    return tokenized_full_prompt
data = data.shuffle().map(generate_and_tokenize_prompt)
print('train dataset and preprocessed loaded.')


# training
output_dir = 'experiments'
training_args = transformers.TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=3,
    logging_steps=1,
    output_dir=output_dir,
    max_steps=30000,
    optim='paged_adamw_8bit',
    lr_scheduler_type='cosine',
    warmup_ratio=.05,
    report_to='tensorboard'
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False) # not using mask language model -> mlm =False
)

model.config.use_cache=False
print('start training.')
trainer.train()
model_output_dir='data/model/sap_sam_2022/adrian_filter/tiiuae/falcon-7b/trained-model'
model.save_pretrained(model_output_dir)
print('model saved.')