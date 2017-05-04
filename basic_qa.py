from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity


import nltk
import json

STOPWORDS = set(stopwords.words('english'))
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()

def remove_stop(sentence):
	new = []
	for word in sentence.split():
		if word in STOPWORDS: continue
		new.append(word)
	return new

def get_bow(document):
	bow = {}
	for word in document.split():
		if word in STOPWORDS: continue
		bow[word] = bow.get(word, 0) + 1
	return bow

def lemmatize(word):
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma

def lemmatize_doc(document):
	output = []
	for word in document:
		output.append(lemmatize(word))
	return output

if __name__ == '__main__':
	documents = []
	qas = []
	BOWs = []
	dict_vectorizer = DictVectorizer()

	with open('QA_test.json') as test_file:
		test = json.load(test_file)

	term_freqs = []
	doc_freqs = []
	for i in range(len(test)):
		doc_freq[i] = {}
		for document in test[i]['sentences']:
			document = word_tokenizer.tokenize(document)
			document = lemmatize(document)
			document = remove_stop(document)
			term_freq = {}
			for word in document:
				term_freq[word] = term_freq.get(word, 0) + 1
			if word in document:
				doc_freq[word] = doc_freq.get(word, 0) + 1
		term_freqs.append(term_freq)
		doc_freqs.append(doc_freq)

	tf_idfs = []
	for i in range(len(term_freqs)):
		tf_idf = {}
		for word, count in term_freqs[i].items():
			tf_idf[word] = count * (1.0 / doc_freqs[i][word])




