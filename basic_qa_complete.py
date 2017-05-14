from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from nltk.tag import StanfordNERTagger
from ner_test04 import parse_docs
from ranking import get_best_answer, get_top_answers, get_question_type

import sent_retrieval as sr
import numpy as np
import nltk
import json
import os
import pprint
pp = pprint.PrettyPrinter(indent=4)

STOPWORDS = set(stopwords.words('english'))
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()

os.environ['CLASSPATH'] = '/usr/share/stanford-ner/stanford-ner.jar'
os.environ['STANFORD_MODELS'] = '/usr/share/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz'

classifier = os.environ.get('STANFORD_MODELS')
jar = os.environ.get('CLASSPATH')
 
st = StanfordNERTagger(classifier,jar)


from tqdm import tqdm
def test_with_dev():
	# load json
	with open('QA_dev.json') as dev_file:
		dev = json.load(dev_file)

	total = 0.0
	match_sentences = 0.0
	match_first_sentence = 0.0
	match_entity = 0.0
	match_first_sentence_entity = 0.0
	match_correct_sentence_entity = 0.0
	match_first_correct_sentence_entity = 0.0
	ranking_failed = 0.0
	match_best_answer = 0.0
	for trial in tqdm(dev):
		# make posting list
		doc_set = trial['sentences']
		no_docs = len(doc_set)
		# NER for all sentences
		entities = parse_docs(doc_set)
		for question in tqdm(trial['qa']):
			# sentence retrieval
			possible_sents = retrieve_sentences(question['question'], doc_set, 20)
			total += 1
			if len(possible_sents) == 0:
				continue

			if question['answer_sentence'] in possible_sents:
				match_sentences += 1

			if question['answer_sentence'] == possible_sents[0]:
				match_first_sentence += 1
			
			# search for entities in possible sents
			matches = []

			# # take only the best match in sentence retrieval
			# matches = [e for e in entities if e[0] == possible_sents[0]]

			# OR...

			# take all sentences into ranking
			for sent in possible_sents:
				matches.extend([e for e in entities if e[0] == sent])

			# determine if correct answer exists in entities
			matches_entities = {m[1] for m in matches}
			first_sentence_entities = {m[1] for m in matches if m[0] == possible_sents[0]}

			retrieval_and_ner_correct = False
			e = None
			for entity in matches:
				if entity[1] == question['answer']:
					e = entity
					break
			
			if e:
				match_entity += 1
				if e[0] == question['answer_sentence']:
					match_correct_sentence_entity += 1
					if possible_sents[0] == e[0]:
						match_first_correct_sentence_entity += 1
						retrieval_and_ner_correct = True

			if question['answer'] in first_sentence_entities:
				match_first_sentence_entity += 1

			if len(matches) == 0:
				continue
			
			# find best answer
			best_match = get_best_answer(
				question['question'],
				matches,
				doc_set,
				possible_sents)

			if best_match[1] == question['answer']:
				# exact match
				match_best_answer += 1
			elif retrieval_and_ner_correct:
				ranking_failed += 1
				top = get_top_answers(
					question['question'],
					matches,
					doc_set,
					possible_sents)
				print "all results:"
				pp.pprint(top)
				print "question:", question['question'].encode('utf-8')
				print "expected:", question['answer'].encode('utf-8')
				print "expected sentence:", doc_set[question['answer_sentence']].encode('utf-8')
				print "actual:", best_match
				print "expected id:", question['answer_sentence']
				print "extracted id:", possible_sents
				question_words = { w.lower() for w in word_tokenizer.tokenize(question['question']) }
				print "predicted question type:", get_question_type(question_words).encode('utf-8')
				# pp.pprint(matches[:5])
				print "\n\n"

	print "% sentence retrieved:", match_sentences / total
	print "% sentence retrieved as first:", match_first_sentence / total
	print "% entity identified:", match_entity / total
	print "% entity identified in first sentence:", match_first_sentence_entity / total
	print "% entity identified in correct sentence:", match_correct_sentence_entity / total
	print "% entity identified in first and correct sentence:", match_first_correct_sentence_entity / total
	print "% above but ranking failed:", ranking_failed / total
	print "% correct best answer:", match_best_answer / total

def escape_csv(answer):
	return answer.replace('"','').replace(',','-COMMA-')

import csv
def make_csv():
	# load json
	with open('QA_test.json') as dev_file:
		dev = json.load(dev_file)

	csv_file = open('output.csv', 'w')
	writer = csv.writer(csv_file)
	writer.writerow(['id', 'answer'])

	for trial in tqdm(dev):
		# make posting list
		doc_set = trial['sentences']
		posting = sr.prepare_doc(doc_set)
		query = process_query(question['question'])
		no_docs = len(doc_set)
		# NER for all sentences
		entities = parse_docs(doc_set)
		for question in tqdm(trial['qa']):
			# sentence retrieval
			possible_sents = eval_query(query, posting, no_docs)[:20]
			if len(possible_sents) == 0:
				writer.writerow( [question['id'], ''] )
				continue
			
			# search for entities in possible sents
			matches = []

			# # take only the best match in sentence retrieval
			# matches = [e for e in entities if e[0] == possible_sents[0]]

			# OR...

			# take all sentences into ranking
			for sent in possible_sents:
				matches.extend([e for e in entities if e[0] == sent])

			if len(matches) == 0:
				writer.writerow( [question['id'], ''] )
				continue
			
			# find best answer
			best_match = get_best_answer(
				question['question'],
				matches,
				doc_set,
				possible_sents)

			writer.writerow( [question['id'], escape_csv(best_match[1]).encode('utf-8')] )

	csv_file.close()


if __name__ == '__main__':
	test_with_dev()
	# make_csv()
