from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

STOPWORDS = set(stopwords.words('english'))

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

if __name__ == '__main__':
	documents = []
	qas = []
	BOWs = []
	dict_vectorizer = DictVectorizer()

	with open('QA_train.json') as training_file:
		training = json.load(training_file)

	print training[0]['sentences'][0]
