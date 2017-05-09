from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tag import StanfordNERTagger

import numpy as np
import nltk
import json
import os

STOPWORDS = set(stopwords.words('english'))
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()

os.environ['CLASSPATH'] = '/usr/share/stanford-ner/stanford-ner.jar'
os.environ['STANFORD_MODELS'] = '/usr/share/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz'

classifier = os.environ.get('STANFORD_MODELS')
jar = os.environ.get('CLASSPATH')
 
st = StanfordNERTagger(classifier,jar)

def remove_stop(sentence):
	new = []
	for word in sentence:
		if not word in STOPWORDS:
			new.append(word)
	return new

# def get_bow(document):
# 	bow = {}
# 	for word in document.split():
# 		if word in STOPWORDS: continue
# 		bow[word] = bow.get(word, 0) + 1
# 	return bow

def get_next_tag (tagged_sent,cur_tag):
    for word, tag in tagged_sent:
        if tag != cur_tag:
            pos = tagged_sent.index((word,tag))
            return pos

def get_continuous_chunks(tagged_sents):
    sents_chunks = []
    for tagged_sent in tagged_sents:
        tagged_sent.append(('end','END'))
        continuous_chunk = []
        sent_not_empty = True


        while sent_not_empty:
            cur_tag = tagged_sent[0][1]
            pos = get_next_tag(tagged_sent,cur_tag)
            chunk = tagged_sent[0:pos]
            continuous_chunk.append(chunk)
            tagged_sent = tagged_sent[pos:]

            if len(tagged_sent) == 1:
                sent_not_empty = False

        sents_chunks.append(continuous_chunk)
    return sents_chunks

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

def get_freqencies(documents):
	term_freqs = []
	doc_freq = {}
	for document in documents:
		document = word_tokenizer.tokenize(document)
		document = lemmatize_doc(document)
		document = remove_stop(document)
		term_freq = {}
		for word in document:
			term_freq[word] = term_freq.get(word, 0) + 1
		for word in list(set(document)):
			doc_freq[word] = doc_freq.get(word, 0) + 1
		term_freqs.append(term_freq)
	return term_freqs, doc_freq

def get_tf_idf(term_freqs, doc_freq):
	tf_idf = []
	words = doc_freq.keys()
	for doc_dict in term_freqs:
		doc_vector = []
		words_in_doc = set(doc_dict.keys())
		for word in words:
			if word in words_in_doc:
				doc_vector.append(doc_dict[word] * 1.0 / doc_freq[word])
			else: doc_vector.append(0)
		tf_idf.append(doc_vector)
	return words, tf_idf

def get_inverted_index(words, tf_idf):
	posting = {}
	for i in range(len(words)):
		posting[words[i]] = posting.get(words[i], [])
		for doc_id in range(len(tf_idf)):
			word_weight = tf_idf[doc_id][i]
			if word_weight != 0:
				posting[words[i]].append((doc_id, word_weight))
	return posting

def process_query(query):
	return remove_stop(lemmatize_doc(word_tokenizer.tokenize(query)))

def prepare_doc(doc_set):
	term_freqs, doc_freq = get_freqencies(doc_set)
	words, tf_idf = get_tf_idf(term_freqs, doc_freq)
	posting = get_inverted_index(words, tf_idf)
	return posting

def eval_query(query, posting, no_docs):
	scores = {}
	for term in query:
		posting_list = posting.get(term, [])
		for (doc_id, weight) in posting_list:
			scores[doc_id] = scores.get(doc_id, 0) + weight
	sorted_scores = sorted(scores.items(), key=lambda x:x[1], reverse=True)
	return [d for d, w in sorted_scores]

def get_most_prob_sent(question, doc_set):
	query = process_query(question)
	posting = prepare_doc(doc_set)
	no_docs = len(doc_set)
	return eval_query(query, posting, no_docs)[:3]



def get_entities(sentences, doc_set):
	entities = []
	for s_id in sentences:
		sentence = doc_set[s_id]
		entities_in_sent = parse_entities(sentence)
		for (entity, entity_type) in entities_in_sent:
			entities.append((s_id, entity, entity_type))
	return entities

def parse_entities(sentence):
	entities = []
	# TODO
	return entities

def get_question_type(question_words):
	"""Determine question type.

	Args:
		question (str): the question as a string

	Returns:
		(str): type of question as a string
	"""
	# TODO more rules
	if "who" in question_words:
		return "PERSON"
	elif "where" in question_words:
		return "LOCATION"
	elif "how many" in question_words:
		return "NUMBER"
	elif "what" in question_words and "year" in question_words:
		return "NUMBER"
	elif "when" in question_words:
		return "NUMBER"
	else:
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

def get_closed_class_words(question_words):
	tagged = nltk.pos_tag(question_words, tagset="universal")
	# consider pronouns, determiners, conjunctions, and prepositions as closed class
	return [p[0] for p in tagged if p[1] in ["PRON", "DET", "CONJ", "ADP"]]

def get_dist_to_question_word(closed_class_words, sentence_words, answer):
	# print closed_class_words
	answer_words = nltk.word_tokenize(answer)
	# cannot proceed if answer word not found in sentence
	for w in answer_words:
		if w not in sentence_words:
			return None
	# get positions of closed class question words
	question_words_pos = []
	for w in closed_class_words:
		for i in range(len(sentence_words)):
			if w == sentence_words[i]:
				question_words_pos.append(i)
	# print question_words_pos
	# cannot proceed if no such closed word in sentence
	if len(question_words_pos) == 0:
		return None
	# (naive way to) find answer position in sentence
	answer_start_pos = sentence_words.index(answer_words[0])
	answer_end_pos = sentence_words.index(answer_words[-1])
	# print "ans start", answer_start_pos
	# print "ans end", answer_end_pos
	# calculate distance and find closest
	dists = [ min(abs(p-answer_start_pos), abs(p-answer_end_pos))
		for p in question_words_pos ]
	# print dists
	return min(dists)

def make_answer_cmp_func(question, doc_set, sentences):
	"""Make comparision function of answers.

	Args:

		doc_set [str]: a list of all answers, indexed by answer id

	Returns:
		comparision funtion of two answers, more relavant answer
		is considered greater
	"""
	question_words = { w.lower() for w in word_tokenizer.tokenize(question) }
	closed_class_words = get_closed_class_words(question_words)
	question_type = get_question_type(question_words)
	def cmp_answer(a, b):
		# First, answers whose content words all appear
		# in the question should be ranked lowest.
		a_words = word_tokenizer.tokenize(a[1].lower())
		b_words = word_tokenizer.tokenize(b[1].lower())
		a_all_appear = contains_all(question_words, a_words)
		b_all_appear = contains_all(question_words, b_words)
		if a_all_appear != b_all_appear:
			return b_all_appear - a_all_appear

		# Second, answers which match the question type
		# should be ranked higher than those that don't;
		a_matches_type = a[2] == question_type
		b_matches_type = b[2] == question_type
		if a_matches_type != b_matches_type:
			return a_matches_type - b_matches_type

		# consider relavance in sentence retrieval
		a_rank = sentences.index(a[0])
		b_rank = sentences.index(b[0])
		if a_rank != b_rank:
			return b_rank - a_rank

		# Third, among entities of the same type, the
		# prefered entity should be the one which is closer
		# in the sentence to a closed-class word from the question.
		a_sent_words = [ w.lower() for w in word_tokenizer.tokenize(doc_set[a[0]]) ]
		b_sent_words = [ w.lower() for w in word_tokenizer.tokenize(doc_set[b[0]]) ]
		a_dist = get_dist_to_question_word(question_words, a_sent_words, a[1].lower())
		b_dist = get_dist_to_question_word(question_words, b_sent_words, b[1].lower())
		if a_dist != b_dist:
			if a_dist == None:
				return -1
			elif b_dist == None:
				return 1
			else:
				return b_dist - a_dist
	return cmp_answer

import pprint
pp = pprint.PrettyPrinter(indent=4)

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
	# answer_scores = []
	# for ans in answers:
	# 	answer_scores.append((ans, get_score(question, ans, doc_set)))
	# return max(answer_scores, key=lambda x: x[1])[0]
	cmp_func = make_answer_cmp_func(question, doc_set, sentences)
	key_func = cmp_to_key(cmp_func)
	return max(answers, key=key_func)

def get_top_answers(question, answers, doc_set, sentences):
	cmp_func = make_answer_cmp_func(question, doc_set, sentences)
	key_func = cmp_to_key(cmp_func)
	return sorted(answers, reverse=True, key=key_func)[:20]

def get_tagged(processed_docs):
	ner_tagged_sents = st.tag_sents(processed_docs)
	tagged_sents = []

	for sent in ner_tagged_sents:
		tagged_sent =[]
		for token,tag in sent:
			if token != '':
				if tag != 'O':
					tagged_sent.append((token,tag))

				else:
					if tag == 'O' and sent.index((token,tag)) != 0 and token[0].isupper():
						tag = 'OTHER'
						tagged_sent.append((token,tag))

					elif tag == 'O' and token.isdigit():
						tag = 'NUMBER'
						tagged_sent.append((token,tag))
					else:
						tagged_sent.append((token,tag))
		tagged_sents.append(tagged_sent)
	return tagged_sents


def parse_docs(doc_set):
    processed_docs = process_doc_ner(doc_set)
    no_docs = len(processed_docs)
    tagged_sents = get_tagged(processed_docs)

    name_entity_list = get_continuous_chunks(tagged_sents)

    doc_ne_pairs = []
    for i in range (0,no_docs):
        name_entity = name_entity_list[i]
        # name_entity_str = [" ".join([token for token, tag in ne]) for ne in name_entity]
        name_entity_pairs = [(i," ".join([token for token, tag in ne]), ne[0][1]) for ne in name_entity]
        for sent_id,entity,tag in name_entity_pairs:
			if tag != 'O':
				if tag == 'ORGANIZATION':
					doc_ne_pairs.append((sent_id,entity,'OTHER'))
				else:
					doc_ne_pairs.append((sent_id,entity,tag))
    return doc_ne_pairs

def process_doc_ner(doc_set):
    # doc as a single sentence
    new_docs = []
    for doc in doc_set:
        doc = word_tokenizer.tokenize(doc)
        new_docs.append(doc)
    return new_docs 

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
	match_best_answer = 0.0
	for trial in tqdm(dev):
		# make posting list
		doc_set = trial['sentences']
		posting = prepare_doc(doc_set)
		no_docs = len(doc_set)
		# NER for all sentences
		entities = parse_docs(doc_set)
		for question in tqdm(trial['qa']):
			# sentence retrieval
			query = process_query(question['question'])
			possible_sents = eval_query(query, posting, no_docs)
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
			if question['answer'] in matches_entities:
				match_entity += 1

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
			elif question['answer'] in first_sentence_entities and possible_sents[0] == question['answer_sentence']:
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
				print "predicted question type:", get_question_type(question['question']).encode('utf-8')
				# pp.pprint(matches[:5])
				print "\n\n"

	print "% sentence retrieved:", match_sentences / total
	print "% sentence retrieved as first:", match_first_sentence / total
	print "% entity identified:", match_entity / total
	print "% entity identified in first sentence:", match_first_sentence_entity / total
	print "% correct best answer:", match_best_answer / total


if __name__ == '__main__':
	test_with_dev()
	
