from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import nltk
import json
# nia-start
import os
from nltk.tag import StanfordNERTagger


os.environ['CLASSPATH'] = '/usr/share/stanford-ner/stanford-ner.jar'
os.environ['STANFORD_MODELS'] = '/usr/share/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz'


classifier = os.environ.get('STANFORD_MODELS')
jar = os.environ.get('CLASSPATH') 
st = StanfordNERTagger(classifier,jar)
# nia -end

STOPWORDS = set(stopwords.words('english'))
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()

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

def test_with_dev(n):
	with open('QA_dev.json') as dev_file:
		dev = json.load(dev_file)

	total = 0.0
	match = 0.0
	for trial in dev:
		doc_set = trial['sentences']
		posting = prepare_doc(doc_set)
		no_docs = len(doc_set)
		for question in trial['qa']:
			query = process_query(question['question'])
			possible_sents = eval_query(query, posting, no_docs)
			total += 1
			if len(possible_sents) == 0:
				continue
			else: sent = possible_sents[:n]
			if sent[0] == question['answer_sentence'] or question['answer_sentence'] in sent:
				match += 1

	return match / total

# nia -start
def prepare_doc(doc_set):
    # doc as a single sentence
    new_docs = []
    for doc in doc_set:
        doc = word_tokenizer.tokenize(doc)
        new_docs.append(doc)
    return new_docs 

def get_next_tag(tagged_sent,cur_tag):
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

def get_tagged(processed_docs):
    no_docs = len(processed_docs)
    ner_tagged_sents = st.tag_sents(processed_docs)
    tagged_sents = []

    for sent in ner_tagged_sents:
        tagged_sent =[]
        for token,tag in sent:
            if tag != 'O':
                tagged_sent.append((token,tag))

            else:
                if tag == 'O' and token[0].isupper():
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
    processed_docs = prepare_doc(sents)
    no_docs = len(processed_docs)
    tagged_sents = get_tagged(processed_docs)

    name_entity_list = get_continuous_chunks(tagged_sents)

    doc_ne_pairs = []
    for i in range (0,no_docs):
        name_entity = name_entity_list[i]
        # name_entity_str = [" ".join([token for token, tag in ne]) for ne in name_entity]
        name_entity_pairs = [(i," ".join([token for token, tag in ne]), ne[0][1]) for ne in name_entity]
        for sent_id,entity,tag in name_entity_pairs:
            if tag == 'ORGANIZATION':
                doc_ne_pairs.append((sent_id,entity,'OTHER'))
            elif tag == 'PERSON' or tag == 'LOCATION' or tag == 'NUMBER':
                doc_ne_pairs.append((sent_id,entity,tag))
    return doc_ne_pairs
# nia -end





def get_question_type(question):
	"""Determine question type.
	Args:
		question (str): the question as a string
	Returns:
		(str): type of question as a string
	"""
	question = " ".join(word_tokenizer.tokenize(question)).lower()
	# TODO more rules
	if "who" in question:
		return "PERSON"
	elif "where" in question:
		return "LOCATION"
	elif "how many" in question:
		return "NUMBER"
	else:
		return "OTHER"

def get_score(question, answer):
	"""Calculate the score of an answer given a question.
	Args:
		question (str): the question as a string
		answers [(str, str, str)]: an answer to the question
			being a 3-tuple of (sentence, entity, entity type)
	Returns:
		(float): score of answer
	"""
	# First, answers whose content words all appear
	# in the question should be ranked lowest.
	ans_words = word_tokenizer.tokenize(answer)
	question_words = set(word_tokenizer.tokenize(question))
	all_appear = True
	for w in ans_words:
		if w not in question_words:
			all_appear = False
			break
	if all_appear:
		return 0.0

	# Second, answers which match the question type
	# should be ranked higher than those that don't;
	if answer[2] != get_question_type(question):
		return 1.0

	# Third, among entities of the same type, the
	# prefered entity should be the one which is closer
	# in the sentence to a closed-class word from the question.
	# TODO

	return 3.0


def get_best_answer(question, answers):
	"""Return the best answer from answers to a question.
	Args:
		question (str): the question as a string
		answers [(str, str, str)]: a list of answers,
			each being a 3-tuple of (sentence, entity, entity type)
	
	Returns:
		(str, str, str): the best answer to the question
	"""
	answer_scores = []
	for ans in answers:
		answer_scores.append((ans, get_score(question, ans)))
	return max(answer_scores, key=lambda x: x[1])[0]


if __name__ == '__main__':
	for n in range(1,20):
print n, test_with_dev(n)