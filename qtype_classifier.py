from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from ner_test04 import *

import numpy as np
import nltk
import json

word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

def lemmatize(word):
	lemma = lemmatizer.lemmatize(word,'v')
	if lemma == word:
		lemma = lemmatizer.lemmatize(word,'n')
	return lemma

def lemmatize_doc(document):
	output = []
	for word in document:
		if word.isalnum():
			output.append(lemmatize(word.lower()))
	return output

def get_training_data():
	answers = []
	questions = []
	with open('QA_train.json') as train_file:
		train_set = json.load(train_file)
		for trail in train_set:
			ans_set = []
			que_set = []
			for qa in trail['qa']:
				ans_set.append(qa['answer'])
				que_set.append(qa['question'])
			answers.extend(ans_set)
			questions.extend(que_set)
	return questions, answers

def tag_answers(answers):
	tag_set = parse_docs(answers)
	return get_tags(answers, tag_set)

def get_tags(ans_set, tag_set):
	tags = []
	tag_idx = 0
	for i in range(len(ans_set)):
		if tag_set[tag_idx]['id'] != i: this_tag = 'OTHER'
		else: this_tag = tag_set[tag_idx]['type']
		while tag_set[tag_idx]['id'] == i:
			if this_tag != tag_set[tag_idx]['type']:
				this_tag = 'OTHER'
			if tag_idx + 1 >= len(tag_set): break
			if tag_set[tag_idx+1]['id'] == i: tag_idx += 1
			else: break
		tags.append('OTHER')
	return tags

def get_que_bow(question):
	q_bow = {}
	question = lemmatize_doc(word_tokenizer.tokenize(question))
	for token in question:
		q_bow[token] = q_bow.get(token, 0) + 1
	return q_bow

def prepare_questions(questions):
	processed_qs = []
	for question in questions:
		q_bow = get_que_bow(question)
		processed_qs.append(q_bow)
	return processed_qs

def get_classifier():
	questions, answers = get_training_data()
	questions = prepare_questions(questions)
	
	ans_tags = tag_answers(answers)
	vectorizer = DictVectorizer()
	dataset = vectorizer.fit_transform(questions)
	classifier = MultinomialNB(2, False, None)
	classifier.fit(dataset, ans_tags)
	return vectorizer, classifier

