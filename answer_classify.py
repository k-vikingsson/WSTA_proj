import nltk
import json
from tqdm import tqdm
from ner_test06 import parse_docs
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sent_retrieval import eval_query, process_query, prepare_doc
from ranking import add_answer_properties, get_question_type, get_open_class_words
from word2vec import get_word2vec_model
import numpy as np
import random

CLASSIFIER_TYPE = LogisticRegression

# w2v = get_word2vec_model()

def get_pos_tag(answer, doc_set):
	tagged = nltk.pos_tag(nltk.word_tokenize(doc_set[answer['id']]))

def make_feature_vector(answer, question_words, question_type, open_class_words, doc_set, sentences):
	added = add_answer_properties(question_words, question_type, open_class_words, answer, doc_set, sentences)
	features = []
	# open_class_words_sum = None
	# for word in open_class_words:
	# 	if word in w2v.wv:
	# 		if open_class_words_sum == None:
	# 			open_class_words_sum = w2v.wv[word]
	# 		else:
	# 			open_class_words_sum += w2v.wv[word]
	# features.extend(open_class_words_sum)
	features.append(float(added['appear_in_question']))
	features.append(float(added['matches_question_type']))
	rank = added['sent_retrieval_rank']
	features.append(-1.0 if rank == None else float(rank))
	dist = added['dist_to_open_words']
	features.append(-1.0 if dist == None else float(dist))
	# TODO add more...
	return features

def train_classifier(sample_trial_size=None, sample_qa_size=None):
	# load json
	with open('QA_train.json') as file:
		dataset = json.load(file)
	
	# take random sample if specified
	if sample_trial_size:
		indices = random.sample(xrange(len(dataset)), sample_trial_size)
		dataset = [dataset[i] for i in indices]

	# get all usable (NER successful) training data
	correct_cases = []
	incorrect_cases = []

	for trial in tqdm(dataset):
		doc_set = trial['sentences']
		posting = prepare_doc(doc_set)
		no_docs = len(doc_set)
		entities = parse_docs(trial['sentences'])
		correct_entities = []

		qa_list = trial['qa']
		# take random sample if specified
		if sample_trial_size:
			indices = random.sample(xrange(len(qa_list)), sample_qa_size)
			qa_list = [qa_list[i] for i in indices]

		for qa in tqdm(qa_list):
			# (lazy) sentence retrieval
			query = process_query(qa['question'])
			question_words = [w.lower() for w in qa['question']]
			question_type = get_question_type(question_words)
			open_class_words = get_open_class_words(question_words)
			possible_sents = eval_query(query, posting, no_docs)
			for e in entities:
				if e['answer'] == qa['answer'] and e['id'] == qa['answer_sentence']:
					correct_cases.append(make_feature_vector(
						e,
						question_words,
						question_type,
						open_class_words,
						doc_set,
						possible_sents
					))
				else:
					incorrect_cases.append(make_feature_vector(
						e,
						question_words,
						question_type,
						open_class_words,
						doc_set,
						possible_sents
					))

	features = np.vstack(( np.array(correct_cases), np.array(incorrect_cases) ))
	outcomes = np.hstack(( np.full((len(correct_cases),), 'y'), np.full((len(incorrect_cases),), 'n') ))
	clf = CLASSIFIER_TYPE()
	clf.fit(features, outcomes)
	return clf

if __name__ == "__main__":
	train_classifier(sample_trial_size=10, sample_qa_size=2)