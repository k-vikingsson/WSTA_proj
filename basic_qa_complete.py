from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from nltk.tag import StanfordNERTagger
import numpy as np
from sent_retrieval import *
from ner_test07 import parse_docs
from ranking import get_best_answer, get_top_answers, get_question_type, get_open_class_words
from evaluation import reciprocal_rank, plot_correct_sent_rank_histogram

import numpy as np
import nltk
import json
import os
import pprint
pp = pprint.PrettyPrinter(indent=4)

word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()

def answer_to_tuple(answer):
	return (
		answer['id'],
		answer['answer'],
		answer['type'],
		answer['start_pos'],
		answer['end_pos'],
		answer['appear_in_question'],
		answer['matches_question_type'],
		answer['dist_to_open_words']
		)

from collections import defaultdict
from tqdm import tqdm
import random
def test_with(filename, sample_trial_size=None, sample_qa_size=None):
	# load json
	with open(filename) as file:
		dataset = json.load(file)
	
	# take random sample if specified
	if sample_trial_size:
		indices = random.sample(xrange(len(dataset)), sample_trial_size)
		dataset = [dataset[i] for i in indices]

	reciprocal_ranks = []
	sentence_rank_freq = defaultdict(int)
	total = 0.0
	num_match_sentences = 0.0
	num_match_first_sentence = 0.0
	num_match_entity = 0.0
	num_match_first_sentence_entity = 0.0
	num_match_correct_sentence_entity = 0.0
	num_match_first_correct_sentence_entity = 0.0
	num_entity_extracted_not_correct_sent = 0.0
	num_ranking_failed = 0.0
	num_correct_answer = 0.0
	for trial in tqdm(dataset):
		# make posting list
		doc_set = trial['sentences']
		posting = prepare_doc(doc_set)
		word_sets = get_word_sets(doc_set)
		no_docs = len(doc_set)
		# NER for all sentences
		all_entities = parse_docs(doc_set)

		qa_list = trial['qa']
		# take random sample if specified
		if sample_trial_size:
			indices = random.sample(xrange(len(qa_list)), sample_qa_size)
			qa_list = [qa_list[i] for i in indices]

		for qa in tqdm(qa_list):
			# sentence retrieval
			query = process_query(qa['question'])
			possible_sents = eval_query(query, posting, word_sets, no_docs)[:20]
			total += 1
			if len(possible_sents) == 0:
				continue

			if qa['answer_sentence'] in possible_sents:
				sentence_rank_freq[possible_sents.index(qa['answer_sentence']) + 1] += 1

			# check sentence retrieval
			sentence_retrieved = qa['answer_sentence'] in possible_sents
			sentence_retrieved_as_first = qa['answer_sentence'] == possible_sents[0]
			num_match_sentences += sentence_retrieved
			num_match_first_sentence += sentence_retrieved_as_first
			
			# search for entities in possible sents
			# # take only the best match in sentence retrieval
			# matches = [e for e in entities if e[0] == possible_sents[0]]

			# OR...
			# take all sentences into ranking
			matches = [e for e in all_entities if e['id'] in set(possible_sents)]
			if len(matches) == 0:
				continue
			
			# search for the correct answer in matches
			correct_entities = [e for e in matches if e['answer'] == qa['answer']]

			# check NER result
			entity_extracted = bool(correct_entities)
			entity_extracted_in_correct_sent = False
			entity_extracted_in_first_sent = False
			entity_extracted_in_first_correct_sent = False
			
			if correct_entities:
				num_match_entity += 1
				entity_extracted_in_first_sent = bool([e for e in correct_entities if e['id'] == possible_sents[0]])
				entity_extracted_in_correct_sent = bool([e for e in correct_entities if e['id'] == qa['answer_sentence']]) 
			
			num_match_correct_sentence_entity += entity_extracted_in_correct_sent
			num_match_first_sentence_entity += entity_extracted_in_first_sent
			retrieval_and_ner_correct = entity_extracted_in_correct_sent and entity_extracted_in_first_sent
			num_match_first_correct_sentence_entity += retrieval_and_ner_correct
			
			# find best answer
			top_answers = get_top_answers(
					qa['question'],
					matches,
					doc_set,
					possible_sents)
			best_match = top_answers[0]

			# if not entity_extracted:
			# 	print "Question:", qa['question'].encode('utf-8')
			# 	print "Answer:", qa['answer'].encode('utf-8')
			# 	print "Correct sentence", doc_set[qa['answer_sentence']].encode('utf-8')
			# 	print "Entities in correct sentence:"
			# 	pp.pprint([e for e in matches if e['id'] == qa['answer_sentence']])
			# 	print "\n\n"

			if entity_extracted and not entity_extracted_in_correct_sent and sentence_retrieved:
				num_entity_extracted_not_correct_sent += 1
				

			# print qa['question']
			# question_words = { w.lower() for w in word_tokenizer.tokenize(qa['question']) }
			# print get_question_type(question_words).encode('utf-8')

			reciprocal_ranks.append(reciprocal_rank(qa['answer'], [a['answer'] for a in top_answers]))

			if best_match['answer'] == qa['answer']:
				# exact match
				num_correct_answer += 1
			elif retrieval_and_ner_correct:
				num_ranking_failed += 1
				print "all results:"
				pp.pprint([answer_to_tuple(a) for a in top_answers])
				print "question:", qa['question'].encode('utf-8')
				print "expected:", qa['answer'].encode('utf-8')
				print "expected sentence:", doc_set[qa['answer_sentence']].encode('utf-8')
				print "actual:", answer_to_tuple(best_match)
				print "expected id:", qa['answer_sentence']
				print "extracted id:", possible_sents
				question_words = [ w.lower() for w in word_tokenizer.tokenize(qa['question']) ]
				print "predicted question type:", get_question_type(question_words).encode('utf-8')
				print "question open class words:", [w.encode('utf-8') for w in get_open_class_words(question_words)]
				# pp.pprint(matches[:5])
				print "\n\n"
				
				

	print "% sentence retrieved:", num_match_sentences / total
	print "% sentence retrieved as first:", num_match_first_sentence / total
	print "% entity identified:", num_match_entity / total
	print "% entity identified but not in correct sentence:", num_entity_extracted_not_correct_sent / total
	print "% entity identified in first sentence:", num_match_first_sentence_entity / total
	print "% entity identified in correct sentence:", num_match_correct_sentence_entity / total
	print "% entity identified in first and correct sentence:", num_match_first_correct_sentence_entity / total
	print "% above but ranking failed:", num_ranking_failed / total
	print "% correct best answer:", num_correct_answer / total
	print "Mean reciprocal rank:", np.mean(reciprocal_ranks)
	plot_correct_sent_rank_histogram(sentence_rank_freq)

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
		posting = prepare_doc(doc_set)
		word_sets = get_word_sets(doc_set)
		no_docs = len(doc_set)
		# NER for all sentences
		entities = parse_docs(doc_set)
		for question in tqdm(trial['qa']):
			# sentence retrieval
			query = process_query(question['question'])
			possible_sents = eval_query(query, posting, word_sets, no_docs)[:20]
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
				matches.extend([e for e in entities if e['id'] == sent])

			if len(matches) == 0:
				# no answer found, write empty answer
				writer.writerow( [question['id'], ''] )
				continue
			
			# find best answer
			best_match = get_best_answer(
				question['question'],
				matches,
				doc_set,
				possible_sents)

			writer.writerow( [question['id'], escape_csv(best_match['answer']).encode('utf-8')] )

	csv_file.close()


if __name__ == '__main__':
	# test_with('QA_train.json')
	# test_with('QA_train.json', sample_trial_size=20, sample_qa_size=10)
	test_with('QA_dev.json')
	# make_csv()
