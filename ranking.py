# from qtype_classifier import get_classifier, lemmatize_doc, get_que_bow

import nltk
import json
from wordnet_func import get_head_word
from syntactic_dist import cfg_path_dist_tagged

word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()
# vectorizer, classifier = get_classifier()

common_measurements = set()
with open("common_measurements.txt") as file:
	for line in file:
		common_measurements.add(line.strip())

common_localities = set()
with open("common_localities.txt") as file:
	for line in file:
		common_localities.add(line.strip())

def get_question_type(question_words):
	"""Determine question type.

	Args:
		question_words ([str]): list of words in question

	Returns:
		(str): type of question as a string
	"""
	# print question_words
	# print lemmas
	# q_bow = {}
	# for word in question_words:
	# 	q_bow[word] = q_bow.get(word, 0) + 1
	# q_vec = vectorizer.transform(q_bow)
	# q_type = classifier.predict(q_vec)
	# return q_type[0]
	# TODO more rules
	if "who" in question_words:
		return "PERSON"
	elif "where" in question_words:
		return "LOCATION"
	elif "how" in question_words and "many" in question_words:
		return "NUMBER"
	elif "when" in question_words and "what" not in question_words:
		return "DATE"
	elif "what" in question_words or "which" in question_words:
		target = get_head_word(question_words)
		try:
			if target[-2:] in ['or','er'] or target[-3:] == 'ist':
				return 'PERSON'
		except: pass
		if target in ["king", "name", "president"]: return "PERSON"
		elif target in ['year']: return "DATE"
		elif target in common_measurements: return "NUMBER"
		elif target in common_localities: return "LOCATION"
	return "OTHER"


def contains_all(items, elems):
	for e in elems:
		if e not in items:
			return False
	return True

# def get_score(question, answer, doc_set, sentences):
# 	"""Calculate the score of an answer given a question.

# 	Args:
# 		question (str): the question as a string
# 		answers [(str, str, str)]: an answer to the question
# 			being a 3-tuple of (sentence, entity, entity type)
# 		doc_set [str]: a list of all answers, indexed by answer id

# 	Returns:
# 		(float): score of answer
# 	"""
# 	# First, answers whose content words all appear
# 	# in the question should be ranked lowest.
# 	ans_words = word_tokenizer.tokenize(doc_set[answer[0]])
# 	question_words = set(word_tokenizer.tokenize(question))
# 	if contains_all(question_words, ans_words):
# 		return 0.0

# 	# Second, answers which match the question type
# 	# should be ranked higher than those that don't;
# 	if answer[2] != get_question_type(question):
# 		return 1.0

# 	# Third, among entities of the same type, the
# 	# prefered entity should be the one which is closer
# 	# in the sentence to a closed-class word from the question.
# 	# TODO

# 	rank = sentences.index(answer[0])
# 	return 1.0 + 100.0/(rank+1)

# def get_closed_class_words(question_words):
# 	tagged = nltk.pos_tag(question_words, tagset="universal")
# 	# consider pronouns, determiners, conjunctions, and prepositions as closed class
# 	return [p[0] for p in tagged if p[1] in ["PRON", "DET", "CONJ", "ADP", "AUX", "NUM", "PART"]]

def get_open_class_words(question_words):
	tagged = nltk.pos_tag(question_words, tagset="universal")
	# consider pronouns, determiners, conjunctions, and prepositions as closed class
	return [p[0] for p in tagged if p[1] in ["ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"]]

def get_cfg_dist_to_question_word(target_words, sentence_words, entity):
	tagged = nltk.pos_tag(sentence_words)
	entity_words = nltk.word_tokenize(entity['answer'])
	return cfg_path_dist_tagged(tagged, target_words[0], entity_words[0])

def get_dist_to_question_word(target_words, sentence_words, entity):
	# get positions of question words
	question_words_pos = []
	for w in target_words:
		for i in range(len(sentence_words)):
			if w == sentence_words[i]:
				question_words_pos.append(i)
	# cannot proceed if no such closed word in sentence
	if len(question_words_pos) == 0:
		return None
	answer_start_pos = entity['start_pos']
	answer_end_pos = entity['end_pos']
	# calculate distance and take sum
	dists = [ min(abs(p-answer_start_pos), abs(p-answer_end_pos))
		for p in question_words_pos ]
	return sum(dists)

def cmp_answer(a, b):
	# First, answers whose content words all appear
	# in the question should be ranked lowest.
	if a['appear_in_question'] != b['appear_in_question']:
		return b['appear_in_question'] - a['appear_in_question']
	# Second, answers which match the question type
	# should be ranked higher than those that don't;
	if a['matches_question_type'] != b['matches_question_type']:
		return a['matches_question_type'] - b['matches_question_type']
	# consider relavance (rank) in sentence retrieval
	if a['sent_retrieval_rank'] != b['sent_retrieval_rank']:
		return b['sent_retrieval_rank'] - a['sent_retrieval_rank']
	# Third, among entities of the same type, the
	# prefered entity should be the one which is closer
	# in the sentence to a closed-class word from the question.
	if a['dist_to_open_words'] != b['dist_to_open_words']:
		if a['dist_to_open_words'] == None:
			return -1
		elif b['dist_to_open_words'] == None:
			return 1
		else:
			return b['dist_to_open_words'] - a['dist_to_open_words']
	return 0

def add_answer_properties(question_words, question_type, open_class_words, answer, doc_set, sentences):
	answer_words = word_tokenizer.tokenize(answer['answer'].lower())
	answer_sent_words = [ w.lower() for w in word_tokenizer.tokenize(doc_set[answer['id']]) ]
	added = dict(answer)
	added['sent_retrieval_rank'] = sentences.index(answer['id'])
	added['appear_in_question'] = contains_all(question_words, answer_words)
	added['matches_question_type'] = answer['type'] == question_type
	added['dist_to_open_words'] = get_dist_to_question_word(open_class_words, answer_sent_words, answer)
	return added

from functools import cmp_to_key
def get_best_answer(question, answers, doc_set, sentences):
	"""Return the best answer from answers to a question.

	Args:
		answers [(str, str, str)]: a list of answers,
			each being a 3-tuple of (sentence, entity, entity type)
		doc_set [str]: a list of all answers, indexed by answer id
		sentences [int]: a list of sentence id's sorted by relavance

	Returns:
		(str, str, str): the best answer to the question
	"""
	question_words = [ w.lower() for w in word_tokenizer.tokenize(question) ]
	question_type = get_question_type(question_words)
	open_class_words = get_open_class_words(question_words)
	answers_added = [
		add_answer_properties(
			question_words,
			question_type,
			open_class_words,
			a,
			doc_set,
			sentences)
		for a in answers
	]
	key_func = cmp_to_key(cmp_answer)
	return max(answers_added, key=key_func)
	# return get_top_answers(question, answers, doc_set, sentences)[0]

def get_top_answers(question, answers, doc_set, sentences, n=None):
	question_words = [ w.lower() for w in word_tokenizer.tokenize(question) ]
	# question_words = question
	question_type = get_question_type(question_words)
	open_class_words = get_open_class_words(question_words)
	answers_added = [
		add_answer_properties(
			question_words,
			question_type,
			open_class_words,
			a,
			doc_set,
			sentences)
		for a in answers
	]
	key_func = cmp_to_key(cmp_answer)
	top = sorted(answers_added, reverse=True, key=key_func)
	if n: top = top[:n]
	return top

if __name__ == '__main__':
	with open('QA_dev.json') as dev_file:
		dev_data = json.load(dev_file)
		questions = [qa['question'] for qa in dev_data[0]['qa']]
		for q in questions:
			print q
			q_bow = get_que_bow(q)
			print get_question_type(q_bow)
			print ''