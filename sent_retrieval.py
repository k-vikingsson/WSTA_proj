from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer

import nltk

STOPWORDS = set(stopwords.words('english'))
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()

def remove_stop(sentence):
	new = []
	for word in sentence:
		if not word in STOPWORDS:
			new.append(word)
	return new

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

def retrieve_sentences(question, doc_set, n=None):
	query = process_query(question)
	posting = prepare_doc(doc_set)
	no_docs = len(doc_set)
	if n == None: return eval_query(query, posting, no_docs)
	return eval_query(query, posting, no_docs)[:n]

# if __name__ == '__main__':

