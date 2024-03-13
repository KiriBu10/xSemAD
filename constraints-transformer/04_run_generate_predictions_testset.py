from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from datasets import load_from_disk
import pickle
import os
import numpy as np 
from tqdm import tqdm
import pandas as pd
from evaluation.utils import generate_prediction_list, calculate_precision_recall, sort_constraints, filter_prediction_list, filter_prediction_list_for_eval,sort_constraints_for_eval
from labelparser.label_utils import constraint_splitter
import torch



model_checkpoint =  "checkpoint-127200/checkpoint-42400"#"checkpoint-118800"
model_name='new/google/flan-t5-small'
dataset='sap_sam_2022/filtered'
max_new_tokens=100
prediction_output_dir = f'data/evaluation/{dataset}/test/{model_name}_{model_checkpoint}/'


# load case names from test set. Only the one bert and svm could handle 
with open('../../ml-semantic-anomaly-dection/evaluation_sap_sam_2022_test_case_names.pkl', 'rb') as f:
    model_case_names = pickle.load(f)


#get all possible constraint types
constraint_type='DECLARE'
constraints_dir = f'data/{dataset}/constraints'
path_to_all_constraint_types_file = os.path.join(constraints_dir,f'ALL_CONSTRAINT_TYPES.{constraint_type}.pkl')
with open(path_to_all_constraint_types_file,'rb') as f:
    all_constraint_types = pickle.load(f)

#load model
model_dir = f"data/model/{dataset}/{model_name}/{model_checkpoint}"
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
#to device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
# 
labels_dir = f'data/{dataset}/constraints_to_log_labels/'
path_to_constraints = f'data/{dataset}/constraints_to_log_labels/'


if not os.path.exists(prediction_output_dir):
    os.makedirs(prediction_output_dir)

for model_case_name in tqdm(model_case_names,desc='make predictions'):
    #print(model_case_name)
    result_list=[]
    path_to_labels = os.path.join(labels_dir,f'{model_case_name}.LABELS.pkl')
    with open(path_to_labels,'rb') as f:
        labels = list(pickle.load(f))
    with open(f'{path_to_constraints}{model_case_name}.CONSTRAINTS.pkl','rb') as f:
        constraints = pickle.load(f)
        all_constraint_types_in_model = list(set([i.split('[')[0] for i in constraints]))
    for c in list(all_constraint_types):
        if c in all_constraint_types_in_model:
            context = c+': <event>'+ '<event>'.join(labels)
            true_c_list = [i for i in constraints if i.startswith(c+ '[') ] 
            true_c_list = sort_constraints(true_c_list, remove_duplicates=True)
            prediction = generate_prediction_list(context,tokenizer,model,30, max_new_tokens=max_new_tokens, device=device)
            prediction = filter_prediction_list_for_eval(model_labels=labels, prediction_c_list=prediction)
            prediction = sort_constraints_for_eval(prediction, remove_duplicates=True)
            result_list.append((c,prediction))
    file_name_path = f'{prediction_output_dir}{model_case_name}.pkl'
    with open(file_name_path, 'wb') as f:
        pickle.dump(result_list, f)
print('DONE!')