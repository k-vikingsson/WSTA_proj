import os
import nltk
import json
from nltk.tag import StanfordNERTagger


os.environ['CLASSPATH'] = '/usr/share/stanford-ner/stanford-ner.jar'
os.environ['STANFORD_MODELS'] = '/usr/share/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz'


classifier = os.environ.get('STANFORD_MODELS')
jar = os.environ.get('CLASSPATH')
 
st = StanfordNERTagger(classifier,jar)
word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()


#sents = ["Rami Eid is studying at Stony Brook University in NY","Anna Brown like Beijing 0709 0303 Shanghai"]

def process_doc_ner(doc_set):
    # doc as a single sentence
    new_docs = []
    for doc in doc_set:
        doc = word_tokenizer.tokenize(doc)
        new_docs.append(doc)
    return new_docs 

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
            if tag == 'ORGANIZATION':
                doc_ne_pairs.append((sent_id,entity,'OTHER'))
            elif tag == 'PERSON' or tag == 'LOCATION' or tag == 'NUMBER':
                doc_ne_pairs.append((sent_id,entity,tag))
    return doc_ne_pairs


with open('QA_dev.json') as dev_file:
    dev = json.load(dev_file)
doc_set = dev[0]['sentences']
print len(doc_set)
test = parse_docs(doc_set)
print test


