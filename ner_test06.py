import os
import nltk
import json
from nltk.tag import StanfordNERTagger
from nltk.corpus import stopwords

# nltk.download('maxent_treebank_pos_tagger')
# nltk.download('averaged_perceptron_tagger')

os.environ['CLASSPATH'] = '/usr/share/stanford-ner/stanford-ner.jar'
os.environ['STANFORD_MODELS'] = '/usr/share/stanford-ner/classifiers/english.muc.7class.distsim.crf.ser.gz'

classifier = os.environ.get('STANFORD_MODELS')
jar = os.environ.get('CLASSPATH')
 
st = StanfordNERTagger(classifier,jar)
word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()
STOPWORDS = set(stopwords.words('english'))
name_list = nltk.corpus.names
names = set([name for file in ('female.txt','male.txt') for name in name_list.words(file)])

#sents = ["Rami Eid is studying at Stony Brook University in NY","Anna Brown like Beijing 0709 0303 Shanghai"]

def process_doc_ner(doc_set):
	# doc as a single sentence
	new_docs = []
	new_docs_pu = []
	for doc in doc_set:
		doc_pu = nltk.word_tokenize(doc)
		doc = word_tokenizer.tokenize(doc)
		new_docs.append(doc)
		new_docs_pu.append(doc_pu)
	return new_docs,new_docs_pu

def get_next_tag (tagged_sent,cur_tag):
	for token, tag in tagged_sent:
		if tag != cur_tag:
			pos = tagged_sent.index((token,tag))
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

def ner_tagger(processed_docs,no_docs):
	ner_tagged_sents = st.tag_sents(processed_docs)
	pos_tagged_sents = nltk.pos_tag_sents(processed_docs)
	tagged_sents = []


	for i in range (0,no_docs):
		# print 'sentence',i
		tagged_sent =[]
		ner_sent = ner_tagged_sents[i]
		pos_sent = pos_tagged_sents[i]
		no_tokens = len(ner_sent)

		for j in range (0, no_tokens):
			token = ner_sent[j][0]
			tag = ner_sent[j][1]

			if token != '':
				if tag != 'O':
					tagged_sent.append((token,tag))

				else:
					if j == 0:
						if token.lower() not in STOPWORDS:
						#token[0].isupper() and token.lower() not in STOPWORDS:
							if token in names:
								#print 'name',token,pos_sent[j][1]

								tag = 'PERSON'
								tagged_sent.append((token,tag))



							elif pos_sent[j][1] in ['NNP', 'NNPS']:
								#print 'nnp',token,pos_sent[j][1]

								tag = 'OTHER'
								tagged_sent.append((token,tag))

					elif token[0].isupper():
						tag = 'OTHER'
						tagged_sent.append((token,tag))

						
					elif pos_sent[j][1] == 'CD':
						#print 'num',token,pos_sent[j][1]

						tag = 'NUMBER'
						tagged_sent.append((token,tag))

					else:
						tagged_sent.append((token,tag))

		tagged_sents.append(tagged_sent)

	return tagged_sents



import pprint
pp = pprint.PrettyPrinter(indent=4)

def subfinder(sent, entity):
	matches = []
	tokens = word_tokenizer.tokenize(entity)
	pl = len(tokens)
	
	for i in range(0,len(sent)):
		if sent[i] == tokens[0] and sent[i:i+pl] == tokens:

			matches.append((i,i+pl-1))
	return matches

def parse_docs(doc_set):
	answers = []
	processed_docs,processed_docs_pu = process_doc_ner(doc_set)
	no_docs = len(processed_docs_pu)

	# iter 01
	tagged_sents_01 = ner_tagger(processed_docs_pu,no_docs)
	name_entity_list_01 = get_continuous_chunks(tagged_sents_01)

	tagged_sents_02 = st.tag_sents(processed_docs)
	name_entity_list_02 = get_continuous_chunks(tagged_sents_02)

	doc_ne_pairs = []
	for i in range (0,no_docs):
		name_entity_01 = name_entity_list_01[i]
		name_entity_02 = name_entity_list_02[i]
		# name_entity_str = [" ".join([token for token, tag in ne]) for ne in name_entity]
		ne_pairs_01= [(" ".join([token for token, tag in ne]), ne[0][1]) for ne in name_entity_01 if ne[0][1] != 'O']
		ne_pairs_02 = [(" ".join([token for token, tag in ne]), ne[0][1]) for ne in name_entity_02 if ne[0][1] != 'O']
		ne_pairs = set(ne_pairs_01 + ne_pairs_02)
		#print ne_pairs
		doc_ne_pairs.extend(ne_pairs)
	
	doc_ne_pairs = list(set(doc_ne_pairs))
	
	for entity,tag in doc_ne_pairs:
		for i in range (0,no_docs):
			sent = processed_docs[i]
			matches = subfinder(sent,entity)
			for match in matches:
				answers.append({'id':i,'answer':entity,'type':tag,'start_pos':match[0],'end_pos':match[1]})

	return answers



	#     for entity,tag in name_entity_pairs:
	#         if tag != 'O':
	#             if tag == 'ORGANIZATION':
	#                 doc_ne_pairs.append({'id':sent_id,'answer':entity,'type':'OTHER','start_pos':start_i,'end_pos':end_i})
	#             elif tag == 'NUMBER':
	#                 text = word_tokenizer.tokenize(entity)
	#                 n = len(text)
	#                 if n != 1:
	#                     end_i = end_i + n - 1
	#                 doc_ne_pairs.append({'id':sent_id,'answer':entity,'type':tag,'start_pos':start_i,'end_pos':end_i})
	#             else:
	#                 doc_ne_pairs.append({'id':sent_id,'answer':entity,'type':tag,'start_pos':start_i,'end_pos':end_i})
	# return doc_ne_pairs


if __name__ == '__main__':
	with open('QA_dev.json') as dev_file:
		dev = json.load(dev_file)

	for i in range (0,5):
		doc_set = dev[i]['sentences']
		pp.pprint(parse_docs(doc_set))
		print ''
