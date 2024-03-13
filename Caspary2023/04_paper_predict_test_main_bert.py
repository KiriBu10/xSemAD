import os

import numpy as np
import pm4py
from collections import Counter

from eval_old import DEVICE_NAME
from sad.transformer.bert_sequence_classifier_model import BertSequenceClassifierModel
from sad.svm.svm_model import SvmModel
#from sad.utils import input_handler
from sad.utils.labelparser.label_utils import sanitize_label
import pickle
from tqdm import tqdm

model_dir= '/ceph/dsola/kb/ml-semantic-anomaly-dection'

log_dir = 'input/sap_sam_2022/filtered/test/noisy_logs'#'input/event_logs'
out_dir = 'output/sap_sam_2022/filtered/reports' # 'output/reports
config = 'BERT'

OUTPUT_TEMPLATE_PRE = 'Anomaly in {TRACE}: {LABEL_1} occurred before {LABEL_2} '
OUTPUT_TEMPLATE_SUCC = 'Anomaly in {TRACE}: {LABEL_1} occurred after {LABEL_2} '


#event_label_key_map = {
#    "BPIC15_5.xes.xml": "activityNameEN",
#}


def run_approach():
    detector_model = load_detector_model(config)
    directory = log_dir
    error_counter=0
    
    for file in tqdm(os.listdir(directory)):
        filename = os.fsdecode(file)
        try:

            if filename.endswith(".xes") or filename.endswith(".mxml"):
                log_file = os.path.join(log_dir, filename)
                df = pm4py.read_xes(log_file)
                print('applying approach on', log_file)
                log = pm4py.convert_to_event_log(df)
                total_anomalies, total_no_anomalies = apply_approach_on_log(detector_model, log, filename)
                print(len(total_anomalies), ' anomalies found')
                print(len(total_no_anomalies), 'no anomalies found')
                print(total_no_anomalies)
                case_name = filename.split('.')[0]
                with open(f'output/sap_sam_2022/filtered/test/{config}/{case_name}.pkl', 'wb') as f:
                    pickle.dump(total_no_anomalies, f)

                #with open(out_dir+"/"+filename.replace(".xes", "") + "_anomaly_report.csv", encoding='utf-8', mode='w') as fp:
                #    fp.write('Anomaly ; freq\n')
                #    for anomaly, count in total_anomalies.items():
                #        fp.write('{}; {}\n'.format(anomaly, count))
        except:
            error_counter+=1
            print('ERROR')
            continue
    print('Done!')
    print('num errors: '+ str(error_counter))


def apply_approach_on_log(detector_model, log, log_name=""):
    seen_variants = {}
    total_anomalies = Counter()
    total_no_anomalies = set()
    label_key = "concept:name"#event_label_key_map.get(log_name, "concept:name")
    for trace in log:
        variant = tuple(sanitize_label(event[label_key]) for event in trace)
        if variant in seen_variants:
            trace_anomalies = seen_variants[variant]
        else:
            trace_anomalies, no_anomalous_pairs = detect_anomalies_in_trace(detector_model, trace, label_key)
            total_no_anomalies.update(no_anomalous_pairs)
            seen_variants[variant] = trace_anomalies
            for precedes, all_followers in trace_anomalies["PRE"].items():
                print(OUTPUT_TEMPLATE_PRE.format(TRACE=trace.attributes["concept:name"], LABEL_1=precedes, LABEL_2=', '.join(sorted(list(all_followers)))))
            for follows, all_predecessors in trace_anomalies["SUCC"].items():
                print(OUTPUT_TEMPLATE_SUCC.format(TRACE=trace.attributes["concept:name"], LABEL_1=follows, LABEL_2=', '.join(sorted(list(all_predecessors)))))
        total_anomalies.update(
            (":".join(OUTPUT_TEMPLATE_PRE.format(TRACE=trace.attributes["concept:name"], LABEL_1=follows, LABEL_2=', '.join(sorted(list(all_predecessors)))).split(":")[1:]) for follows, all_predecessors in trace_anomalies["SUCC"].items()))
        total_anomalies.update(
            (":".join(OUTPUT_TEMPLATE_SUCC.format(TRACE=trace.attributes["concept:name"], LABEL_1=precedes, LABEL_2=', '.join(sorted(list(all_followers)))).split(":")[1:]) for precedes, all_followers in trace_anomalies["PRE"].items()))
    return total_anomalies, total_no_anomalies


def simplify_output(anomalous_pairs, simplify=True):
    simplified_output = {"PRE": {}, "SUCC": {}}
    delete_from_pre = set()
    if not simplify:
        for pair in anomalous_pairs:
            if pair[0] not in simplified_output["PRE"]:
                simplified_output["PRE"][pair[0]] = {pair[1]}
    else:
        for pair in anomalous_pairs:
            if pair[0] not in simplified_output["PRE"]:
                simplified_output["PRE"][pair[0]] = {pair[1]}
            else:
                simplified_output["PRE"][pair[0]].add(pair[1])
        for pair in anomalous_pairs:
            if pair[1] not in simplified_output["SUCC"]:
                simplified_output["SUCC"][pair[1]] = {pair[0]}
            else:
                simplified_output["SUCC"][pair[1]].add(pair[0])
        for label in simplified_output["PRE"]:
            if label in simplified_output["SUCC"] and len(simplified_output["SUCC"][label]) < len(simplified_output["PRE"][label]):
                del simplified_output["SUCC"][label]
            elif label in simplified_output["SUCC"] and len(simplified_output["PRE"][label]) <= len(simplified_output["SUCC"][label]):
                delete_from_pre.add(label)
        for label in delete_from_pre:
            del simplified_output["PRE"][label]
    return simplified_output


def detect_anomalies_in_trace(detector_model, trace, label_key):
    label_sequence = [event[label_key] for event in trace]
    label_pairs = extract_label_pairs(label_sequence)
    # detect anomalies
    anomalous_pairs, no_anomalous_pairs = detect_anomalous_pairs(detector_model, label_pairs)
    # post-process for trace (?)
    anomalous_pairs = simplify_output(anomalous_pairs, simplify=True)
    return anomalous_pairs, no_anomalous_pairs


def detect_anomalous_pairs(detector_model, label_pairs):
    anomalous_pairs = set()
    no_anomalous_pairs=set()
    preds, probs = detector_model.make_predictions(np.asarray(list(label_pairs), dtype="object"))
    print(preds)
    for i, label_pair in enumerate(label_pairs):
        # pred = detector_model.make_single_prediction(label_pair) SLOW!
        anomalous_pairs.add(label_pair) if preds[i] == 1 else None
        no_anomalous_pairs.add(label_pair) if preds[i] == 0 else None
    return anomalous_pairs, no_anomalous_pairs


def extract_label_pairs(label_sequence):
    sub_sequences = split_into_subtraces(label_sequence)
    label_pairs = set()
    for sub_seq in sub_sequences:
        #label_pairs.update([(sanitize_label(sub_seq[i]), sanitize_label(sub_seq[j])) for i in range(len(sub_seq)) for j in range(i+1, len(sub_seq))])
        label_pairs.update([(sub_seq[i], sub_seq[j]) for i in range(len(sub_seq)) for j in range(i+1, len(sub_seq))]) ###################################################change this back 
    return label_pairs


def split_into_subtraces(label_sequence):
    subtraces = []
    current_subtrace = []
    for label in label_sequence:
        if label not in current_subtrace:
            current_subtrace.append(label)
        else:
            subtraces.append(current_subtrace)
            current_subtrace = [label]
    subtraces.append(current_subtrace)
    return subtraces


def load_detector_model(config):
    if config == "BERT":
        bert_sequence_classifier = BertSequenceClassifierModel("bert-base-uncased", DEVICE_NAME, 5e-5, 500)
        bert_sequence_classifier.load(model_dir, "bert-base-uncased")
        return bert_sequence_classifier
    elif config == "SVM":
        svm_classifier = SvmModel("rbf", 0, 2)
        return svm_classifier.load_model(model_dir)
    return True


if __name__ == "__main__":
    run_approach()


