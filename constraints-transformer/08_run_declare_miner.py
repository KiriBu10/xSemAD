import pandas as pd
import os 
from tqdm import tqdm
import pickle
from Declare4Py.ProcessModels.DeclareModel import DeclareModel
from Declare4Py.ProcessMiningTasks.Discovery.DeclareMiner import DeclareMiner
from Declare4Py.D4PyEventLog import D4PyEventLog
from evaluation.utils import sort_constraints

#### DECLARE MINER: Generate declarative constraints for test set ####
prediction_output_dir = 'data/sap_sam_2022/filtered/DECLAREMINER/testset/constraints/' # CHANGE IT! - Directory for saving prediction output from DECLAREMINER
path_to_noisy_test_logs='../../../ml-semantic-anomaly-dection/ml-semantic-anomaly-dection/input/sap_sam_2022/filtered/test/noisy_logs/' # CHANGE IT! - Path to the directory containing noisy test logs

# get test case names
d = pd.read_pickle('evaluation_sap_sam_2022_test_label2constraint_for_comparison_set_new.pkl') # CHANGE IT! 
test_case_names = d.case_name.unique()


constraints_of_interest = ['Alternate Precedence',
 'Alternate Response',
 'Alternate Succession',
 'Choice',
 'Co-Existence',
 'End',
 'Exclusive Choice',
 'Init',
 'Precedence',
 'Response',
 'Succession']

if not os.path.exists(prediction_output_dir):
    os.makedirs(prediction_output_dir)

for model_case_name in tqdm(test_case_names, desc='declarative process mining'):
    # load noisy log
    event_log = D4PyEventLog(case_name="case:concept:name")
    log_path = f'{path_to_noisy_test_logs}{model_case_name}.xes'
    event_log.parse_xes_log(log_path)
    
    # PREDICTION: declarative process discovery
    discovery = DeclareMiner(log=event_log, consider_vacuity=False, min_support=0.2, itemsets_support=0.9, max_declare_cardinality=3) # standard parameter: DeclareMiner(log=event_log, consider_vacuity=False, min_support=0.2, itemsets_support=0.9, max_declare_cardinality=3)
    discovered_model: DeclareModel = discovery.run()
    constraints_declare_miner = [i.split(' | |')[0] for i in discovered_model.serialized_constraints if i.split('[')[0] in constraints_of_interest]
    pred_pairs_temp = sort_constraints(constraints_declare_miner, remove_duplicates=True)
    file_name_path = f'{prediction_output_dir}{model_case_name}.pkl'
    with open(file_name_path, 'wb') as f:
        pickle.dump(pred_pairs_temp, f)
