from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from ner import *

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
	answers_sents = []
	questions = []
	sentences = []
	total_s = 0
	count = 0
	with open('QA_train.json') as train_file:
		train_set = json.load(train_file)
		for trail in train_set:
			if count < 150: count += 1
			else: break
			ans_set = []
			que_set = []
			ans_sents = []
			sent_set = trail['sentences']
			for qa in trail['qa']:
				ans_set.append(qa['answer'])
				ans_sents.append(qa['answer_sentence']+total_s)
				que_set.append(qa['question'])
			total_s += len(sent_set)
			answers.extend(ans_set)
			questions.extend(que_set)
			sentences.extend(sent_set)
			answers_sents.extend(ans_sents)
	return questions, sentences, answers, answers_sents

def tag_sents(sentences):
	return parse_docs(sentences)

def classify_sents(tagged, answers):
	# print tagged
	classified = np.empty(len(answers), dtype=object)
	# organize tagged
	tagged_sents = []
	for i in range(len(answers)):
		tagged_sents.append([])
	this_sent = []
	for entity in tagged:
		tagged_sents[entity['id']].append(entity)
	# initiaize classified
	for i in range(len(classified)):
		classified[i] = (i, None)
	# finalize
	for i in range(len(answers)):
		for entity in tagged_sents[i]:
			if entity['answer'] == answers[i]:
				classified[i] = (i, entity['type'])

	return classified

def filter_train(questions, classes):
	resulting_questions = []
	resulting_classes = []
	for i in range(len(classes)):
		if classes[i][1] != None:
			resulting_questions.append(questions[i])
			resulting_classes.append(classes[i][1])
	return resulting_questions, resulting_classes

def get_que_bow(question, words):
	q_bow = {}
	question = lemmatize_doc(word_tokenizer.tokenize(question))
	iters = len(question)
	for i in range(iters):
		if question[i] not in words: continue
		q_bow[question[i].lower()] = q_bow.get(question[i].lower(), 0) + 1
	return q_bow

def prepare_questions(questions, words):
	processed_qs = []
	for question in questions:
		q_bow = get_que_bow(question,words)
		processed_qs.append(q_bow)
	return processed_qs

def get_all_bow(sentences):
	words = {}
	for sent in sentences:
		sent = lemmatize_doc(word_tokenizer.tokenize(sent))
		for word in sent:
			words[word] = words.get(word, 0) +  1
	return words

def get_classifier():
	questions, sentences, answers, asentids = get_training_data()
	newsentences = [sentences[i] for i in asentids]
	all_bow = get_all_bow(sentences)
	words = set([word for word, count in all_bow.items() if count > 20])
	tagged_sents = tag_sents(newsentences)

	classified_sents = classify_sents(tagged_sents, answers)
	ques, classes = filter_train(questions, classified_sents)

	questions = prepare_questions(ques, words)
	
	vectorizer = DictVectorizer()
	dataset = vectorizer.fit_transform(questions)
	classifier = MultinomialNB(2, False, None)
	classifier.fit(dataset, classes)

	return vectorizer, classifier

