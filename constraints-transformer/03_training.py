import os
import pandas as pd
import pickle
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback,SwitchTransformersEncoderModel
import transformers
import torch


train_dataset='sap_sam_2022/filtered'
dataset_for_training_dir = f'data/{train_dataset}/forTraining'
training_dataset_filename = 'training'
path_to_training_dataset = os.path.join(dataset_for_training_dir,training_dataset_filename)
dataset = load_from_disk(path_to_training_dataset)

from transformers import AutoTokenizer

#model_checkpoint = 'google/switch-base-8'#'google/flan-t5-small'#"t5-small"
model_checkpoint ='google/flan-t5-small'# 't5-base'#'google/flan-t5-base'#'t5-small'#'google/flan-t5-small'
model_dir = f"data/model/{train_dataset}/{model_checkpoint}"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.add_special_tokens({'additional_special_tokens': ['<event>']})
def preprocess_function(tokenizer, example):
    input =  example["context"]
    target =  example["target"]
    model_inputs = tokenizer(input, 
                    padding = "max_length",
                    max_length = 350,
                    return_tensors = "pt",
                    truncation=True)
    target_encoding  = tokenizer(target,
                    padding = "max_length",
                    max_length = 300, 
                    return_tensors = "pt",
                    truncation=True) 
    labels = target_encoding.input_ids
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels#["input_ids"]
    return model_inputs
tokenized_dataset = dataset.map(lambda example: preprocess_function(tokenizer, example), batched=True)

# training
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=20)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
#model = SwitchTransformersEncoderModel.from_pretrained(model_checkpoint)
device = "cuda" if torch.cuda.is_available() else "cpu"
print('---------------------')
print(device)
print('---------------------')
model = model.to(device)
model.resize_token_embeddings(len(tokenizer))
transformers.logging.set_verbosity_info()
seq2seq_args = Seq2SeqTrainingArguments(
    output_dir = model_dir,
    learning_rate = 4e-5, # 3e-4,
    lr_scheduler_type = "constant",
    evaluation_strategy = "steps",
    eval_steps=400,
    save_strategy = "steps",
    save_steps=400,
    num_train_epochs=12,
    weight_decay=0.015,
    report_to="tensorboard",
    load_best_model_at_end = True,
    per_device_train_batch_size = 64,
    per_device_eval_batch_size = 64
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model,
    seq2seq_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[early_stopping_callback],
)
trainer.train()
if torch.cuda.is_available():
    model = model.to("cpu")
