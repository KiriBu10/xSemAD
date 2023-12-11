import sys
import os
import json
import pickle
import gc
import signal
from tqdm import tqdm
import logging
import pm4py
from datetime import datetime
from pm4py.algo.analysis.woflan import algorithm as woflan

sys.path.append('../')

from conversion.json2petrinet import JsonToPetriNetConverter
from bpmn2constraints.bpmnconstraints.script import compile_bpmn_diagram
import conversion.petrinetanalysis as pna
from config import config_preprocessing_param


def is_english_bpmn(path_to_directory:str, json_file:str)->bool:
    # Checks whether considered json file is an English BPMN 2.0 model according to meta file
    json_file = json_file.replace(".json", ".meta.json")
    with open(os.path.abspath(path_to_directory) + "/" + json_file, 'r') as f:
        data = f.read()
        json_data = json.loads(data)
    mod_language = json_data['model']['modelingLanguage']
    nat_language = json_data['model']['naturalLanguage']
    if mod_language == "bpmn20" and nat_language == "en":
        if 'Task' in json_data['revision']['elementCounts']:
            if json_data['revision']['elementCounts']['Task']>2: # 
                return True
    return False

def get_only_englisch_bpmn_models(json_dir:str, json_files:list)->list:
    return [model for model in json_files if is_english_bpmn(json_dir,model)]   

def convert_jsons_to_petri(json_dir:str, json_files:list, target_petri_dir:str):
    converter = JsonToPetriNetConverter()
    #json_files = [f for f in os.listdir(json_dir) if f.endswith(".json") and not f.endswith("meta.json")]
    json_files.sort()
    print("Total number of json files:", len(json_files))
    success = 0
    failed = 0

    if not os.path.exists(target_petri_dir):
        os.makedirs(target_petri_dir)

    for json_file in tqdm(json_files,desc='Converting Json to Petri-Nets'):
        case_name = os.path.basename(json_file).split('.')[0]
        try:
            # Load and convert json-based BPMN into Petri net
            net, initial_marking, final_marking = converter.convert_to_petri_net(
                os.path.join(json_dir, json_file))
            pnet_file = os.path.join(target_petri_dir, case_name + ".pnser")
            pickle.dump((net, initial_marking, final_marking), open(pnet_file, 'wb'))
            success += 1
        except:
            #print("WARNING: Error during conversion from bpmn to Petri net.")
            failed += 1
        if (success + failed) % 50 == 0:
            gc.collect()
    print(success + failed, "jsons done. Succes: ", success, "failed: ", failed)
    print("Run completed.") 

def generate_logs_from_petri_sers(timeout:int,petri_dir_en:str, target_dir:str, target_dir_no_loops:str, target_dir_lables:str):
    def alarm_handler(signum, frame):
        raise Exception("timeout")
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if not os.path.exists(target_dir_no_loops):
        os.makedirs(target_dir_no_loops)
    if not os.path.exists(target_dir_lables):
        os.makedirs(target_dir_lables)
    pnet_ser_files = [f for f in os.listdir(petri_dir_en) if f.endswith(".pnser")]
    pnet_ser_files.sort()
    #pnet_ser_files = pnet_ser_files[start_index:end_index]
    #print("Total number of pnet files:", len(pnet_ser_files))
    success = 0
    done = 0

    for ser_file in tqdm(pnet_ser_files, desc='Creating Event-Logs'):
        case_name = os.path.basename(ser_file).split('.')[0]
        filepath = os.path.join(petri_dir_en, ser_file)
        if os.path.getsize(filepath) > 0:
            net, initial_marking, final_marking = pickle.load(open(filepath, 'rb'))
            try:
                signal.signal(signal.SIGALRM, alarm_handler)
                signal.alarm(timeout)
                is_sound = woflan.apply(net, initial_marking, final_marking, parameters={woflan.Parameters.RETURN_ASAP_WHEN_NOT_SOUND: True,
                                                                 woflan.Parameters.PRINT_DIAGNOSTICS: False,
                                                                 woflan.Parameters.RETURN_DIAGNOSTICS: False})
                # is_relaxed_sound = soundness.check_relaxed_soundness_net_in_fin_marking(net, initial_marking,
                #                                                                         final_marking)
                #print('ASDFASDFA')
                #signal.alarm(0)
            except Exception:
                #print("WARNING: Time out during soudness checking.")
                is_sound=False
                signal.alarm(0)

            if is_sound:
                #print('model is sound. Generating traces')
                log = pna.playout_net(net, initial_marking, final_marking, timeout, extensive=False)
                log_no_loops = pna.create_log_without_loops(log)
                if len(log) > 1:
                    df_log = pm4py.convert_to_dataframe(log)
                    df_log['concept:name']=df_log['concept:name'].str.replace('  ',' ').str.lower()
                    avg_label_names = df_log['concept:name'].str.len().mean()
                    if avg_label_names > 4: # filter out models with labels like this: aa, bb, cc, dd, ...
                        xes_file = os.path.join(target_dir, case_name + ".xes")
                        pm4py.write_xes(log, xes_file)

                        xes_file2 = os.path.join(target_dir_no_loops, case_name + ".xes")
                        pm4py.write_xes(log_no_loops, xes_file2)
                        #print(f"Saved as model {xes_file}")

                        #save labels
                        label_file = os.path.join(target_dir_lables, case_name + ".pkl")
                        with open(label_file, 'wb') as file:
                            pickle.dump(set(df_log['concept:name'].unique()),file)
                        success += 1


        done += 1
        if done % 25 == 0:
            gc.collect()
    msg=f"Number of converted (sound) models: {success} / {done}"
    print(msg)
    print("Run completed.")





def generate_constraints_from_json(json_dir:str, target_constraint_dir:str, constraint_type:str, json_names_to_process:list=None):
    if not os.path.exists(target_constraint_dir):
        os.makedirs(target_constraint_dir)
    
    if not json_names_to_process:
        json_files = os.listdir(json_dir)
    else:
        json_files = json_names_to_process

    all_constraint_types = set()

    for json_file in tqdm(json_files, desc='generate constraints'):
        case_name = os.path.basename(json_file).split('.')[0]
        path_to_file = os.path.join(json_dir,f'{case_name}.json')
        try:
            constraints = compile_bpmn_diagram(path_to_file, constraint_type, skip_named_gateways=True) # DECLARE, SIGNAL, LTLF, 
            constraints = list(set(constraints))
            for constraint in constraints:
                all_constraint_types.add(constraint.split('[')[0])
        except TypeError:
            continue
        constraints_file = os.path.join(target_constraint_dir, case_name + f'.{constraint_type}'+".pkl")
        with open(constraints_file, 'wb') as file:
            pickle.dump(constraints,file)
    path_to_all_constraints_types_file = os.path.join(target_constraint_dir,f'ALL_CONSTRAINT_TYPES.{constraint_type}.pkl')
    with open(path_to_all_constraints_types_file, 'wb') as file:
        pickle.dump(all_constraint_types,file)

print('preprocessing configs:')
print(config_preprocessing_param)

json_dir = config_preprocessing_param['json_dir']
json_files = [f for f in os.listdir(json_dir) if f.endswith(".json") and not f.endswith("meta.json")]
#msg='#models in '+json_dir +': '+ str(len(json_files))
#print(msg)
#json_files_bpmn_en = get_only_englisch_bpmn_models(json_dir,json_files)
json_files_bpmn_en=json_files
petri_nets_dir=config_preprocessing_param['petri_nets_dir']
#convert_jsons_to_petri(json_dir,json_files_bpmn_en,petri_nets_dir)

timeout=config_preprocessing_param['timeout']
logs_dir=config_preprocessing_param['logs_dir']
logs_no_loops_dir=config_preprocessing_param['logs_no_loops_dir']
target_dir_lables=config_preprocessing_param['target_dir_lables']
#generate_logs_from_petri_sers(timeout=timeout,petri_dir_en=petri_nets_dir, target_dir=logs_dir, target_dir_no_loops=logs_no_loops_dir,target_dir_lables=target_dir_lables)


target_constraint_dir=config_preprocessing_param['target_constraint_dir']
constraint_type=config_preprocessing_param['constraint_type']
json_names_to_process = os.listdir(logs_dir)
generate_constraints_from_json(json_dir=json_dir,target_constraint_dir=target_constraint_dir,constraint_type=constraint_type,json_names_to_process=json_names_to_process)

print('DONE!')