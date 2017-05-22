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
	answers_sents = []
	questions = []
	sentences = []
	total_s = 0
	count = 0
	with open('QA_train.json') as train_file:
		train_set = json.load(train_file)
		for trail in train_set:
			if count < 100: count += 1
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
		# print entity
		# if entity['id'] == len(tagged_sents):
		# 	this_sent.append(entity)
		# else:
		# 	tagged_sents.append(this_sent)
		# 	this_sent = []
		# 	this_sent.append(entity)
	# initiaize classified
	for i in range(len(classified)):
		classified[i] = (i, None)
	# print len(tagged_sents), len(answers)
	# finalize
	for i in range(len(answers)):
		# print newsentences[i]
		# print [sent['answer'] for sent in tagged_sents[i]]
		# print answers[i]
		# print ''
		for entity in tagged_sents[i]:
			if entity['answer'] == answers[i]:
				classified[i] = (i, entity['type'])
		# if answers[i] in [s['answer'] for s in tagged_sents[i]]:
		# 	if classified[i][1] == None:
		# 		classified[i] = (i, answers[i]['type'])
		# 	elif classified[answers[i]['id']][1] != answers[i]['type']:
		# 		classified[i] = (i, 'OTHER')

	return classified

def filter_train(questions, classes):
	resulting_questions = []
	resulting_classes = []
	for i in range(len(classes)):
		if classes[i][1] != None:
			resulting_questions.append(questions[i])
			resulting_classes.append(classes[i][1])
	return resulting_questions, resulting_classes

def get_que_bow(question):
	q_bow = {}
	question = lemmatize_doc(word_tokenizer.tokenize(question))
	for token in question:
		q_bow[token.lower()] = q_bow.get(token.lower(), 0) + 1
	return q_bow

def prepare_questions(questions):
	processed_qs = []
	for question in questions:
		q_bow = get_que_bow(question)
		processed_qs.append(q_bow)
	return processed_qs

def get_classifier():
# if __name__ == '__main__':
	questions, sentences, answers, asentids = get_training_data()
	newsentences = [sentences[i] for i in asentids]
	# print len(newsentences), len(answers)
	tagged_sents = tag_sents(newsentences)
	# for tag in tagged_sents:
	# 	print tag
	classified_sents = classify_sents(tagged_sents, answers)
	ques, classes = filter_train(questions, classified_sents)

	# for i in range(len(ques)):
	# 	print ques[i]
	# 	print classes[i]
	# 	print ''



	questions = prepare_questions(ques)
	
	# ans_tags = tag_answers(answers)
	vectorizer = DictVectorizer()
	dataset = vectorizer.fit_transform(questions)
	classifier = MultinomialNB(2, False, None)
	classifier.fit(dataset, classes)

	##### TEST WITH FIRST TRAIL OF TEST #####
	# with open('QA_test.json') as test_file:
	# 	test_json = json.load(test_file)

	# qa = test_json[0]['qa']
	# qs = [x['question'] for x in qa]
	# qs = prepare_questions(qs)


	return vectorizer, classifier

