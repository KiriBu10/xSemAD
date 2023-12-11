from more_itertools import chunked
import numpy as np
from labelparser.label_utils import constraint_splitter

def prediction_cleaning(p):
    p = p.replace("<pad>","").replace("</s>","")
    p = p.strip()
    p = " ".join(p.split()) 
    return p

def collect_scores(predictions,scores):
    predictions_dict = dict()
    maxLen = 1
    for p,s in zip(predictions,scores):
        if p in predictions_dict:
            predictions_dict[p].append(s)
            if len(predictions_dict[p]) > maxLen:
                maxLen = len(predictions_dict[p])
        if p not in predictions_dict:
            predictions_dict[p] = [float(s)]
    return predictions_dict, maxLen

def __ranking_max(predictions, scores,num_recommondations):
    predictions_dict, maxLen = collect_scores(predictions,scores)
    # save the activities together with their confidences in a list but such that every activity has the same number of confidences (add 0's)
    # also sort the confidences per activity
    predictions_list = []
    if maxLen == 1:
        predictions_list = [(k,v) for k,v in predictions_dict.items()]
    else:
        for k,v in predictions_dict.items():
            v.sort(reverse=True)
            while len(v) < maxLen:
                v.append(0)
            predictions_list.append((k,v))
    # now sort the activites according to their confidences
    ranking = sorted(predictions_list, key=lambda tup: tuple(map(lambda i: tup[1][i],list(range(len(tup[1]))))), reverse=True)
    # if two or more activites have the same confidence, sort them according to their label
    ranking = sorted(ranking, key=lambda tup: (tup[1][0],tup[0]), reverse=True)
    # reduce the sorted list to the top10 activities with their maximum confidences
    return ranking[:num_recommondations]

def generate_prediction_list(input_sequences, tokenizer, model, num_recommondations, max_new_tokens=200, device='cpu'):
    inputs = tokenizer(input_sequences,return_tensors='pt',padding=True).to(device)
    sample_output = model.generate(
                max_new_tokens = max_new_tokens,
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                num_return_sequences=num_recommondations,
                num_beams=num_recommondations,
                no_repeat_ngram_size = 50, # no word repetitions
                early_stopping=True, #True,
                return_dict_in_generate=True,
                output_scores=True
                )
    predictions=[]
    scores=[]
    for preds_sequence,scores_sequence,sequence in zip(chunked(sample_output["sequences"].cpu(),num_recommondations),
                                                       chunked(sample_output["sequences_scores"].cpu(),num_recommondations),input_sequences):
        preds_sequence = tokenizer.batch_decode(preds_sequence, skip_special_tokens=False)
        preds_sequence = [prediction_cleaning(p) for p in preds_sequence]
        #preds_sequence = list(set([p for p in preds_sequence if p!=""]))[:num_recommondations]
        #if len(preds_sequence)<10:
        #   print("length of rec list"+str(len(preds_sequence)))
        predictions.extend(preds_sequence)
        scores.extend(scores_sequence)
    scores = np.exp(scores)#softmax(scores)#np.exp(scores)#1/(1+np.exp([-s for s in scores])) #np.exp(scores)
    recommendations_with_score = [(r,round(float(s[0]),3)) for r,s in __ranking_max(predictions, scores,num_recommondations)]
    return recommendations_with_score


def calculate_precision_recall(true_list, prediction_list):
    intersection_num = len(list(set(true_list).intersection(set(prediction_list))))
    
    recall = intersection_num/len(true_list)
    if len(prediction_list)!=0:
        precision = intersection_num/len(prediction_list)
        return precision, recall
    else:
        precision = 0
        return precision, recall

def sort_constraints(constraints_list,correct_spelling=False, remove_duplicates=True):
    result=[]
    for c in constraints_list:
        constraint_type, labels = constraint_splitter(c, correct_spelling=correct_spelling)
        if labels == None:
            return result
        labels = [i.strip() for i in labels]
        if constraint_type.lower() in ['coice', 'co-existence','exclusive choice']:
            labels.sort()
        result.append(f'{constraint_type}['+ ', '.join(labels) +']')
    if remove_duplicates:
        result = list(set(result))
    return result

def filter_prediction_list(model_labels,prediction_c_list):
    result=[]
    for prediction in prediction_c_list:
        c, labels = constraint_splitter(prediction, correct_spelling=False)
        if labels != None:
            labels = [i.strip() for i in labels]
            if set(labels).issubset(model_labels):
                result.append(prediction)
    return result

def filter_prediction_list_for_eval(model_labels,prediction_c_list):
    result=[]
    for prediction in prediction_c_list:
        c, labels = constraint_splitter(prediction[0], correct_spelling=False)
        if labels != None:
            labels = [i.strip() for i in labels]
            if set(labels).issubset(model_labels):
                result.append((prediction[0],prediction[1]))
    return result

def sort_constraints_for_eval(constraints_list,correct_spelling=False, remove_duplicates=True):
    result=[]
    for c in constraints_list:
        constraint_type, labels = constraint_splitter(c[0], correct_spelling=correct_spelling)
        if labels == None:
            return result
        labels = [i.strip() for i in labels]
        if constraint_type.lower() in ['coice', 'co-existence','exclusive choice']:
            labels.sort()
        result.append((f'{constraint_type}['+ ', '.join(labels) +']', c[1]))
    if remove_duplicates:
        result = list(set(result))
    return result