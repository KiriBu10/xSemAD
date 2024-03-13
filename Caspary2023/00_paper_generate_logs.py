import pm4py
from datetime import datetime
from pm4py.algo.analysis.woflan import algorithm as woflan
import os
import signal
from tqdm import tqdm
import pickle
import gc
import sad.utils.petrinetanalysis as pna

data_type='test'


def generate_logs_from_petri_sers(model_ids,timeout:int,petri_dir_en:str, target_dir:str, target_dir_no_loops:str, target_dir_lables:str):
    def alarm_handler(signum, frame):
        raise Exception("timeout")
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if not os.path.exists(target_dir_no_loops):
        os.makedirs(target_dir_no_loops)
    if not os.path.exists(target_dir_lables):
        os.makedirs(target_dir_lables)
    pnet_ser_files = [f for f in os.listdir(petri_dir_en) if (f.endswith(".pnser") and (f.split('.')[0] in model_ids))]
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
                #signal.signal(signal.SIGALRM, alarm_handler)
                #signal.alarm(timeout)
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
                #signal.alarm(0)

            if is_sound:
                #print('model is sound. Generating traces')
                log = pna.playout_net(net, initial_marking, final_marking, timeout, extensive=True)
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

from datasets import load_from_disk

train_dir = f'../constraintCheckingUsingLLM/constraints-transformer/data/sap_sam_2022/filtered/forTraining/training/{data_type}/'
data_train = load_from_disk(train_dir)
model_ids = list(set(data_train['id']))
#len(model_ids)
timeout=30

petri_nets_dir='../constraintCheckingUsingLLM/constraints-transformer/data/sap_sam_2022/filtered/petri_nets/'
logs_dir=f'input/sap_sam_2022/filtered/{data_type}/logs/'
logs_no_loops_dir=f'input/sap_sam_2022/filtered/{data_type}/logs_no_loops/'
target_dir_lables=f'input/sap_sam_2022/filtered/{data_type}/labels/'
generate_logs_from_petri_sers(model_ids,timeout=timeout,petri_dir_en=petri_nets_dir, target_dir=logs_dir, target_dir_no_loops=logs_no_loops_dir,target_dir_lables=target_dir_lables)