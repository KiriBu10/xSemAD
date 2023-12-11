import os
import pandas as pd
import pickle
from tqdm import tqdm
import random 
from sklearn.model_selection import train_test_split
from datasets import load_from_disk, Dataset, DatasetDict
from labelparser.label_utils import get_relevant_constraints

dataset_name='sap_sam_2022/filtered'
constraint_type='DECLARE'
labels_dir=f'data/{dataset_name}/labels'
constraints_dir = f'data/{dataset_name}/constraints'
logs_dir = f'data/{dataset_name}/logs'
dataset_for_training_dir = f'data/{dataset_name}/forTraining'
training_dataset_filename = 'training'
shuffle_labels=True
constraints_to_log_labels_dir=f'data/{dataset_name}/constraints_to_log_labels'

process_labels_files = [i.split('.')[0] for i in os.listdir(labels_dir)]
process_constraints_files = [i.split('.')[0] for i in os.listdir(constraints_dir) if i.endswith(f".{constraint_type}.pkl")]
process_logs_files=[i.split('.')[0] for i in os.listdir(logs_dir)]
case_names = list(set(process_labels_files).intersection(process_constraints_files,process_logs_files))

# get all constraint types
path_to_all_constraint_types_file = os.path.join(constraints_dir,f'ALL_CONSTRAINT_TYPES.{constraint_type}.pkl')
with open(path_to_all_constraint_types_file,'rb') as f:
    all_possible_constraint_types = pickle.load(f)

if not os.path.exists(constraints_to_log_labels_dir):
    os.makedirs(constraints_to_log_labels_dir)
#create trainingsdataset
id_list, context_list, target_list=[],[],[]
for case_name in tqdm(case_names, desc='process processes'):
    path_to_label_file = os.path.join(labels_dir,f'{case_name}.pkl')
    path_to_constraint_file = os.path.join(constraints_dir,f'{case_name}.{constraint_type}.pkl')
    with open(path_to_label_file,'rb') as f:
        model_labels = list(pickle.load(f))
    with open(path_to_constraint_file,'rb') as f:
        model_constraints = pickle.load(f)
    model_labels, model_constraints = get_relevant_constraints(model_labels=model_labels, model_constraints=model_constraints)#this list contains only constraints which labels are similar to the model labels extracted from the event logs
    
    label_file = os.path.join(constraints_to_log_labels_dir, case_name + ".LABELS.pkl")
    with open(label_file, 'wb') as file:
        pickle.dump(model_labels, file)
    constraints_file = os.path.join(constraints_to_log_labels_dir, case_name + ".CONSTRAINTS.pkl")
    with open(constraints_file, 'wb') as file:
        pickle.dump(model_constraints, file)
    
    for c_type in all_possible_constraint_types:
        c_type_in_process = False
        for d_constraint in model_constraints:
            if d_constraint.startswith(c_type):
                c_type_in_process=True
                id_list.append(case_name)
                if shuffle_labels:
                    random.shuffle(model_labels)
                context = " <event> " + " <event> ".join(model_labels)
                context_list.append(f'{c_type}: {context}')
                target_list.append(d_constraint)
        if not c_type_in_process:
            id_list.append(case_name)
            if shuffle_labels:
                random.shuffle(model_labels)
            context = " <event> " + " <event> ".join(model_labels)
            context_list.append(f'{c_type}: {context}')
            target_list.append(f'{c_type}: None')
data = {'id':id_list,'context':context_list,'target':target_list}
df = pd.DataFrame(data)


process_ids = list(df['id'].unique())
ids_train, ids_test = train_test_split(process_ids,test_size=0.3,random_state=22) 
ids_validate, ids_test = train_test_split(ids_test,test_size=0.5,random_state=22) 
df_train = df[df['id'].isin(ids_train)]
df_test = df[df['id'].isin(ids_test)]
df_validate = df[df['id'].isin(ids_validate)]

dataset = DatasetDict()
dataset['train'] = Dataset.from_pandas(df_train.sample(frac=1,random_state=15).reset_index(drop=True))
dataset['validation'] = Dataset.from_pandas(df_validate.sample(frac=1,random_state=15).reset_index(drop=True))
dataset['test'] = Dataset.from_pandas(df_test.sample(frac=1,random_state=15).reset_index(drop=True))


path_to_training_dataset = os.path.join(dataset_for_training_dir,training_dataset_filename)
dataset.save_to_disk(path_to_training_dataset)

print('Done!')