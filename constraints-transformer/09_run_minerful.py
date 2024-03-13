import pandas as pd
import os 
from tqdm import tqdm
import pickle
from Declare4Py.ProcessModels.DeclareModel import DeclareModel
from Declare4Py.ProcessMiningTasks.Discovery.DeclareMiner import DeclareMiner
from Declare4Py.D4PyEventLog import D4PyEventLog
from evaluation.utils import sort_constraints
import subprocess

# to run this visit https://github.com/cdc08x/MINERful

minerful_jar_path = '../../../MINERful/MINERful.jar' # CHANGE IT! - Minerful jar file 
output_path = 'data/sap_sam_2022/filtered/MINERFUL/testset/constraints/' # CHANGE IT! - Directory for saving prediction output from DECLAREMINER
noisy_log_dir ='../../../ml-semantic-anomaly-dection/ml-semantic-anomaly-dection/input/sap_sam_2022/filtered/test/noisy_logs/' # CHANGE IT! - Path to the directory containing noisy test logs

# load test case names
d = pd.read_pickle('evaluation_sap_sam_2022_test_label2constraint_for_comparison_set_new.pkl') # CHANGE IT! 
test_case_names = d.case_name.unique()


def run_minerful_with_parameters(minerful_jar_path, xes_log_path,case_name, output_path):
    current_dir = os.getcwd() 
    minerful_jar_abs_path = os.path.abspath(os.path.join(current_dir, minerful_jar_path))
    xes_log_abs_path = os.path.abspath(os.path.join(current_dir, xes_log_path))
    output_dir = os.path.abspath(os.path.join(current_dir, output_path))
    
    # Ensure paths are correctly resolved
    #print(f"Resolved MINERful JAR Path: {minerful_jar_abs_path}")
    #print(f"Resolved XES Log Path: {xes_log_abs_path}")
    
    lib_path = os.path.join(os.path.dirname(minerful_jar_abs_path), 'lib', '*')  # for Windows
    
    java_path = r'C:\Program Files\Common Files\Oracle\Java\javapath\java.exe' 
    # Java command
    command = [
        java_path,
        '-cp',
        f"{lib_path};{minerful_jar_abs_path}",  # Ensure this is correct for Windows
        'minerful.MinerFulMinerStarter',
        '-iLF', xes_log_abs_path,
        '-oJSON', os.path.join(output_dir, f'{case_name}.json'),
        '-s', '0.95',
        '-c', '0.25',
        '-i', '0.125',
        '-prune', 'none'
    ]
    #print("Executing command:", ' '.join(command))

    try:
        result = subprocess.run(command)#, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error running MINERful: {e}"
    except FileNotFoundError as e:
        return f"File not found error: {e}. Ensure Java is installed and paths are correct."
    



for case_name in tqdm(test_case_names, desc='declarative process mining'):
    xes_log_path = f'{noisy_log_dir}{case_name}.xes'
    result = run_minerful_with_parameters(minerful_jar_path, xes_log_path,case_name, output_path)