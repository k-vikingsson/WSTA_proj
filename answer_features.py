import nltk
import json
from tqdm import tqdm
from ner_test06 import parse_docs

def get_pos_tag(answer, doc_set):
    tagged = nltk.pos_tag(nltk.word_tokenize(doc_set[answer['id']]))

def make_feature_vector(answer):
    pass

def train_classifier():
    correct_cases = []
    # get all usable (NER successful) training data
    with open('QA_train.json') as train_file:
        train_set = json.load(train_file)
        for trial in tqdm(train_set):
            entities = parse_docs(trial['sentences'])
            correct_entities = []
            for qa in trial['qa']:
                for e in entities:
                    if e['answer'] == qa['answer'] and e['id'] == qa['answer_sentence']:
                        correct_cases.append({'entity': e, 'question': qa['question'], 'sentences': trial['sentences']})
    pass # TODO