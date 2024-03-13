from os.path import exists

import numpy as np
import pandas as pd

import sad.utils.input_handler as input_handler
import pickle
import time
from sklearn.metrics import classification_report
import sad.utils.output_handler as output_handler
from sad.config import config_dic_svm
from sad.transformer.bert_sequence_classifier_model import BertSequenceClassifierModel
from sad.svm.svm_model import SvmModel
from sad.utils import pca
from sad.utils.train_test import generate_train_test_split

RANDOM_SEED = 254
DEVICE_NAME = "cuda:1"


def get_pairs_not_in_training_data(train_data, test_data, test_data_classes):
    res = [i for i, (test_pair, clz) in enumerate(zip(test_data, test_data_classes)) if test_pair not in train_data]
    return res


def evaluate_configuration(KERNEL, VECTOR_SIZE, C_VALUE, DEG):
    train_test_split = pickle.load(open("sap_sam_2022_filtered_train_split.p", "rb"))
    reports = []
    reports_bo = []
    reports_new = []
    training_times = []
    for config_key in config_dic_svm.keys():
        if config_key > 1:
            continue

        #train_data_plain = train_test_split["train_data"]
        #test_data_plain = train_test_split["test_data"]
        #test_data_classes_plain = train_test_split["test_data_classes"]

        train_data = train_test_split["train_data_1"]
        train_data_classes = train_test_split["train_data_classes_1"]
        #test_data = train_test_split["test_data_1"]
        #test_data_classes = train_test_split["test_data_classes_1"]
        #test_data_bo = train_test_split["test_data_bo_1"]
        #test_data_classes_bo = train_test_split["test_data_classes_bo_1"]
        #test_data_new = get_pairs_not_in_training_data(train_data_plain, test_data_plain, test_data_classes_plain)
        #print(f"Test data unseen: {len(test_data_new)}")
        #print(f"Test data: {len(test_data)}")

        train_data = pca.reduce_vector_size(
            train_data,
            VECTOR_SIZE,
        )
        #test_data = pca.reduce_vector_size(
        #    test_data,
        #    VECTOR_SIZE,
        #)

        #if len(test_data) > 1000000:
        #    test_data = test_data[0:1000000]
        #    test_data_classes = test_data_classes[0:1000000]

        #if len(test_data_bo) > 1000000:
        #    test_data_bo = test_data_bo[0:1000000]
        #    test_data_classes_bo = test_data_classes_bo[0:1000000]

        svm_model = SvmModel(KERNEL, DEG, C_VALUE)
        (
            tr_report,
            val_predictions,
            training_time,
        ) = svm_model.train_and_validate_svm(train_data, train_data_classes, [], [])

        svm_model.save_model(".")

        #report, predictions = svm_model.test_svm(test_data, test_data_classes)

        #report_bo, predictions_bo = svm_model.test_svm(
        #    test_data_bo, test_data_classes_bo
        #)

        #y_pred_new = [y_p for i, y_p in enumerate(predictions) if i in test_data_new]
        #y_test_new = [y_t for i, y_t in enumerate(test_data_classes) if i in test_data_new]
        #y_pred_new_bo = [y_p for i, y_p in enumerate(predictions_bo) if i in test_data_new]
        #y_test_new_bo = [y_t for i, y_t in enumerate(test_data_classes_bo) if i in test_data_new]
        #print(f"Pred data new: {len(y_pred_new)}")
        #report_new = classification_report(y_test_new, y_pred_new, output_dict=True)
        #report_new_bo = classification_report(
        #    y_test_new_bo, y_pred_new_bo, output_dict=True
        #)

        #report = {k: {k2: round(v2, 2) for k2, v2 in v.items()} for k, v in report.items() if type(v) == dict}
        #report_bo = {
        #    k: {k2: round(v2, 2) for k2, v2 in v.items()} for k, v in report_bo.items() if type(v) == dict
        #}
        #report_new = {
        #    k: {k2: round(v2, 2) for k2, v2 in v.items()} for k, v in report_new.items() if type(v) == dict
        #}
        #report_new_bo = {
        #    k: {k2: round(v2, 2) for k2, v2 in v.items()} for k, v in report_new_bo.items() if type(v) == dict
        #}
        #report_dict = {
        #    "full data": report,
        #    "bo data": report_bo,
        #    "unseen data": report_new,
        #    "unseen bo data": report_new_bo,
        #}
        #report_df = pd.DataFrame(report_dict).transpose()
        #report_df.to_csv(f"report_SVM_K:{KERNEL},C:{C_VALUE},V:{VECTOR_SIZE},D:{DEG}.csv")
        #reports.append(report)
        #reports_bo.append(report_bo)
        #reports_new.append(report_new)
        #training_times.append(training_time)
        #print("Test report")
        #print(report)

    #return (
    #    reports,
    #    reports_bo,
    #    reports_new,
    #    training_times,
    #)
    print('SVM model saved ')


def evaluate_configuration_2(learning_rate, warmup_steps, bert_model_name):
    class_names = ["no anomaly", "anomaly"]
    collect_samples = {0: set(), 1: set()}
    collect_samples_bo = {0: set(), 1: set()}
    train_test_split = pickle.load(open("sap_sam_2022_filtered_train_split.p", "rb"))
    reports = []
    reports_bo = []
    reports_new = []
    training_times = []
    for config_key in config_dic_svm.keys():
        if config_key > 1:
            continue
        train_data = train_test_split["train_data"]
        train_data_classes = train_test_split["train_data_classes"]
        #test_data = train_test_split["test_data"]
        #test_data_classes = train_test_split["test_data_classes"]
        #test_data_bo = train_test_split["test_data_bo"]
        #test_data_classes_bo = train_test_split["test_data_classes_bo"]
        #test_data_new = get_pairs_not_in_training_data(train_data, test_data, test_data_classes)
        #print(f"Test data unseen: {len(test_data_new)}")
        #print(f"Test data: {len(test_data)}")

        bert_sequence_classifier = BertSequenceClassifierModel(
            bert_model_name, DEVICE_NAME, learning_rate, warmup_steps
        )
        bert_sequence_classifier.initialize_train_data(
            train_data, train_data_classes
        )
        #bert_sequence_classifier.initialize_test_data(test_data, test_data_classes)
        #bert_sequence_classifier.initialize_test_data_bo(
        #    test_data_bo, test_data_classes_bo
        #)

        training_start = time.time()
        training_instances = len(train_data)

        val_acc, val_loss = bert_sequence_classifier.trainer()
        training_end = time.time()
        bert_sequence_classifier.save_model(".")

        training_time = training_end - training_start
        
        print(
            f"Training instances: {training_instances}, training time: {training_time}"
        )

        #(
        #    y_review_texts,
        #    y_pred,
        #    y_pred_probs,
        #    y_test,
        #) = bert_sequence_classifier.evaluate_model_on_test_data()

        #(
        #    y_review_texts,
        #    y_pred_bo,
        #    y_pred_probs,
        #    y_test_bo,
        #) = bert_sequence_classifier.evaluate_model_on_test_data_bo()

        #y_pred_new = [y_p for i, y_p in enumerate(y_pred) if i in test_data_new]
        #y_test_new = [y_t for i, y_t in enumerate(y_test) if i in test_data_new]
        #y_pred_new_bo = [y_p for i, y_p in enumerate(y_pred_bo) if i in test_data_new]
        #y_test_new_bo = [y_t for i, y_t in enumerate(y_test_bo) if i in test_data_new]
        #print(f"Pred data new: {len(y_pred_new)}")

        #for i in range(len(test_data)):
        #    if i < len(y_pred) and y_pred[i] == test_data_classes[i]:
        #        collect_samples[test_data_classes[i]].add(tuple(test_data[i]))
        #for i in range(len(test_data_bo)):
        #    if i < len(y_pred_bo) and y_pred_bo[i] == test_data_classes_bo[i]:
        #        collect_samples_bo[test_data_classes_bo[i]].add(tuple(test_data_bo[i]))
        #report = classification_report(
        #    y_test, y_pred, target_names=class_names, output_dict=True
        #)
        # cut off the anything after the first two decimals in the report

        #report_bo = classification_report(
        #    y_test_bo, y_pred_bo, target_names=class_names, output_dict=True
        #)

        #report_new = classification_report(
        #    y_test_new, y_pred_new, target_names=class_names, output_dict=True
        #)

        #report_new_bo = classification_report(
        #    y_test_new_bo, y_pred_new_bo, target_names=class_names, output_dict=True
        #)
        #report = {k: {k2: round(v2, 2) for k2, v2 in v.items()} for k, v in report.items() if type(v) == dict}
        #report_bo = {
        #    k: {k2: round(v2, 2) for k2, v2 in v.items()} for k, v in report_bo.items() if type(v) == dict
        #}
        #report_new = {
        #    k: {k2: round(v2, 2) for k2, v2 in v.items()} for k, v in report_new.items() if type(v) == dict
        #}
        #report_new_bo = {
        #    k: {k2: round(v2, 2) for k2, v2 in v.items()} for k, v in report_new_bo.items() if type(v) == dict
        #}
        #report_dict = {
        #    "full data": report,
        #    "bo data": report_bo,
        #    "unseen data": report_new,
        #    "unseen bo data": report_new_bo,
        #}
        #report_df = pd.DataFrame(report_dict).transpose()
        #report_df.to_csv(f"report_BERT{learning_rate}-{warmup_steps}-{bert_model_name}.csv")

        #print(report)
        #reports.append(report)
        #reports_bo.append(report_bo)
        #reports_new.append(report_new)
        #training_times.append(training_time)
        #records = []
        #for entry in collect_samples[1]:
        #    records.append({"text": entry, "label": 1})
        #for entry in collect_samples[0]:
        #    records.append({"text": entry, "label": 0})
        #df = pd.DataFrame(records)
        #df.to_csv(f"collect_samples_{config_key}.csv", index=False)
        #for entry in collect_samples_bo[1]:
        #    records.append({"text": entry, "label": 1})
        #for entry in collect_samples_bo[0]:
        #    records.append({"text": entry, "label": 0})
        #df = pd.DataFrame(records)
        #df.to_csv(f"collect_samples_bo_{config_key}.csv", index=False)
    #return (
    #    reports,
    #    reports_bo,
    #    reports_new,
    #    training_times,
    #)
    print('training BERT done!')


def evaluate_model1():
    model_configurations = [
        (
            "rbf",
            0,
            600,
            pow(2, 1),
        )
    ]
    model_iteration = 1
    for mc in model_configurations:
        reports, reports_bo, reports_new, training_times = evaluate_configuration(mc[0], mc[2], mc[3], mc[1])
        training_times.extend(training_times)
        model_iteration = model_iteration + 1


def evaluate_model2():
    model_configurations = [
        ("bert-base-uncased", 5e-5, 500)
    ]
    model_iteration = 1
    for mc in model_configurations:
        reports, reports_bo, reports_new, training_times = evaluate_configuration_2(mc[1], mc[2], mc[0])
        model_iteration = model_iteration + 1
        output_handler.report_to_csv(
            reports,
            f"M2_Eval_C{model_iteration}",
            input_handler.output_dir,
        )
        output_handler.report_to_csv(
            reports_bo,
            f"M2_Eval_C{model_iteration}-BO",
            input_handler.output_dir,
        )
        output_handler.report_to_csv(
            reports_new,
            f"M2_Eval_C{model_iteration}-Unseen",
            input_handler.output_dir,
        )


def train_and_save():
    """
    Fine-tune and save BERT model on full training data for use on custom data
    """
    data = pickle.load(open("train_test_split.p", "rb"))
    train_data = data["train_data"]
    train_data_classes = data["train_data_classes"]
    bert_sequence_classifier = BertSequenceClassifierModel(
        "bert-base-uncased", DEVICE_NAME, 5e-5, 500
    )
    bert_sequence_classifier.initialize_train_data(
        train_data, train_data_classes
    )
    training_start = time.time()
    training_instances = len(train_data)
    bert_sequence_classifier.trainer()
    bert_sequence_classifier.save_model(".")
    training_end = time.time()
    training_time = training_end - training_start
    print(f"Training instances: {training_instances}, training time: {training_time}")


if __name__ == "__main__":
    if not exists("sap_sam_2022_filtered_train_split.p"):
        generate_train_test_split()
    #evaluate_model1()
    
    evaluate_model2()
