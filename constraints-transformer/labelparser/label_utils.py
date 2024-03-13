import re
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob


lemmatizer = WordNetLemmatizer()

NON_ALPHANUM = re.compile('[^a-zA-Z]')
CAMEL_PATTERN_1 = re.compile('(.)([A-Z][a-z]+)')
CAMEL_PATTERN_2 = re.compile('([a-z0-9])([A-Z])')


def sanitize_label(label):
    # handle some special cases
    label = label.replace('\n', ' ').replace('\r', '')
    label = label.replace('(s)', 's').replace('&', 'and').strip()
    label = re.sub(' +', ' ', label)
    # turn any non alphanumeric characters into whitespace
    label = NON_ALPHANUM.sub(' ', label)
    label = label.strip()
    # remove single character parts
    label = " ".join([part for part in label.split() if len(part) > 1])
    # handle camel case
    label = _camel_to_white(label)
    # make all lower case
    label = label.lower()
    return label


def split_label(label):
    label = label.lower()
    result = re.split('[^a-zA-Z]', label)
    return result


def _camel_to_white(label):
    label = CAMEL_PATTERN_1.sub(r'\1 \2', label)
    return CAMEL_PATTERN_2.sub(r'\1 \2', label)


def split_and_lemmatize_label(label):
    words = split_label(label)
    lemmas = [lemmatize_word(w) for w in words]
    return lemmas


def lemmatize_word(word):
    lemma = lemmatizer.lemmatize(word, pos='v')
    lemma = re.sub('ise$', 'ize', lemma)
    return lemma

def correct_spelling_of_label(label):
    sentence = TextBlob(label)
    result = sentence.correct()
    return str(result).strip()  

def constraint_splitter(constraint:str, correct_spelling=True)-> (str, list):
    if len(constraint.split('['))==1:
        constraint_type = constraint.split(':')[0]
        labels=None
        return constraint_type, labels
    constraint_type = constraint.split('[')[0]
    try:
        labels = re.search(r'\[[A-Z 1-9a-z,_]*\]', constraint).group()
        labels = re.sub("\[|\]", "", labels)
    except AttributeError:
        labels=None
        return constraint_type, labels
    if correct_spelling:
        labels = [correct_spelling_of_label(i) for i in labels.split(',')]
    else:
        labels = labels.split(',')
    return constraint_type, labels

def constraint_builder(constraint_type:str, labels:list)->str: #reverse function of constraint_splitter
    return f'{constraint_type}['+ ', '.join(labels) +']'


def get_relevant_constraints(model_labels, model_constraints):
    model_labels = [correct_spelling_of_label(i) for i in model_labels]
    #print(model_labels)
    final_constraint_list = [] #this list contains only constraints which labels are similar to the model labels extracted from the event logs
    for constraint in model_constraints:
        constraint_type, constraint_labels = constraint_splitter(constraint)
        if set(constraint_labels).issubset(set(model_labels)):
            final_constraint_list.append(f'{constraint_type}['+ ', '.join(constraint_labels) +']')
    return model_labels, final_constraint_list