from gensim.models import Word2Vec
from tqdm import tqdm
import json
import nltk
from nltk.corpus import brown, movie_reviews, treebank
from itertools import chain

def make_elems_lower(sents):
    for s in sents:
        yield [w.lower() for w in s]

# brown_sents = make_elems_lower(brown.sents())
# mr_sents = make_elems_lower(movie_reviews.sents())
# tb_sents = make_elems_lower(treebank.sents())

# print len(training_sents)
# print len(brown.sents())
# print len(movie_reviews.sents())
# print len(treebank.sents())

def get_word2vec_model():
    training_sents = []
    with open('QA_train.json') as train_file:
        train_set = json.load(train_file)
        for trial in tqdm(train_set):
            for s in trial['sentences']:
                training_sents.append([w.lower() for w in nltk.word_tokenize(s)])
            for qa in trial['qa']:
                training_sents.append([w.lower() for w in nltk.word_tokenize(qa['question'])])
    print "start training word2vec"
    return Word2Vec(training_sents, iter=50)