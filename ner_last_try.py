import os
import nltk
import json
from nltk.tag import StanfordNERTagger
import Tkinter


#nltk.download('words')

os.environ['CLASSPATH'] = '/usr/share/stanford-ner/stanford-ner.jar'
os.environ['STANFORD_MODELS'] = '/usr/share/stanford-ner/classifiers/english.muc.7class.distsim.crf.ser.gz'


classifier = os.environ.get('STANFORD_MODELS')
jar = os.environ.get('CLASSPATH')
 
st = StanfordNERTagger(classifier,jar)
word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()


# sent1 = 'The NASCAR Sprint Cup Series holds two exhibition events annually - the Sprint Unlimited, held at Daytona International Speedway at the start of the season, and the NASCAR Sprint All-Star Race, held at Charlotte Motor Speedway midway through the season.'
# sent2 = 'J.K. Rowling, the English writer, lives in the South of Boston Common and she has 13,231 books. '
# #s1 = word_tokenizer.tokenize(sent)
# s1 = nltk.word_tokenize(sent1)
# s2 = nltk.word_tokenize(sent2)

# # ner = st.tag(s2)
# pos1 = nltk.pos_tag(s1)
# pos2 = nltk.pos_tag(s2)
# test = nltk.ne_chunk(pos)



def process_doc_ner(doc_set):
    # doc as a single sentence
    new_docs = []
    # new_docs_pu = []
    for doc in doc_set:
        # doc_pu = nltk.word_tokenize(doc)
        doc = nltk.word_tokenize(doc)
        new_docs.append(doc)
        # new_docs_pu.append(doc_pu)
    return new_docs


def pos_tagger(processed_docs):
	pos_tagged_sents = nltk.pos_tag_sents(processed_docs)
	return pos_tagged_sents


def get_np(doc_set):
	processed_docs = process_doc_ner(doc_set)
	pos_tagged_sents = pos_tagger(processed_docs)
	no_docs = len(processed_docs)

	cp = nltk.RegexpParser('''
		NP: {<DT>?<JJ>*<NN>}
		NP: {<DT>?<NNP>+<IN>?<NNP>+}
		NP: {<DT>?<NNP>+}
		NP: {<CD>+}''')


	trees = cp.parse_sents(pos_tagged_sents)

	entities = []
	entities_dict = []
	i = 0

	for tree in trees:
		sub_e = []  # empty sentence
		sub_e_d = []
		for subtree in tree.subtrees():
			if subtree.label() == 'NP':
				et = subtree.leaves()
				entity = " ".join([token for token, tag in et])
				entity_d = [token for token, tag in et]
				sub_e.append(entity)
				sub_e_d.append(entity_d)
		entities.append((i,sub_e))
		entities_dict.append((i,sub_e_d))
		i = i + 1

	print 'parse_done'
	doc_ner_pairs = []

	for i,sub_e_d in entities_dict:
		if sub_e_d != []:
			sub_ner_entities = st.tag_sents(sub_e_d)
			print 'loop-1'

			for ner_entity in sub_ner_entities:
				no_tokens = len(ner_entity)
				# print ner_entity
				print 'sub_loop'

				entity = " ".join([token for token, tag in ner_entity])

				no_tokens = len(ner_entity)
				for j in range (0,no_tokens):
					print 'entity check'
					token = ner_entity[j][0]
					tag = ner_entity[j][1]

					if tag != 'O':
						doc_ner_pairs.append({'id':i,'answer':entity,'type':tag})
						break

					elif tag == 'O' and nltk.pos_tag(token) == 'CD':
						doc_ner_pairs.append({'id':i,'answer':entity,'type':'NUMBER'})
						break

					elif tag == 'O' and j == no_tokens - 1:
						doc_ner_pairs.append({'id':i,'answer':entity,'type':'OTHER'})



	return doc_ner_pairs,entities








if __name__ == '__main__':
    with open('QA_dev.json') as dev_file:
        dev = json.load(dev_file)

    # for i in range (0,1):
    doc_set = dev[0]['sentences']
    # entities = get_np
    answers = dev[0]['qa']
        # re = parse_docs(doc_set)
        # for r in re:
        #     print r['pos_sent']
    total = len(answers)
    match = 0
    match_dict = 0
    entities_dict,entities = get_np(doc_set)


    for answer in answers:
    	answer_id = answer['answer_sentence']
    	question = answer['question']
    	a = answer['answer']


    	for entity in entities_dict:
    		if entity['id'] == answer_id and entity['answer'] == a:
    			match_dict = match_dict + 1
    			print question,
    			print ''
    			print entity['type'],a
    			print ' '


    	for sent_id,entity_list in entities:
    		if sent_id == answer_id:
    			if a in entity_list:
    				match = match + 1
    				# print question,
    				# print entity_list,a,sent_id
    				# print ' '
    				# for entity in entities_dict:
    				# 	if entity['id'] == answer_id:
    				# 		print entity 


print match_dict
print total
print float(match_dict)/float(total)

print match
print total
print float(match)/float(total)

# for re in result:
# 	t =  next(re)
# 	print t.flatten()

# print pos

# def process_doc_ner(doc_set):
#     # doc as a single sentence
#     new_docs = []
#     for doc in doc_set:
#         doc = word_tokenizer.tokenize(doc)
#         new_docs.append(doc)
#     return new_docs 

# with open('QA_dev.json') as dev_file:
#     dev = json.load(dev_file)
        
# for i in range (0,5):
#     doc_set = dev[i]['sentences']
#     no_docs = len(doc_set)
#     processed_docs = process_doc_ner(doc_set)
#     ner_tagged_sents = st.tag_sents(processed_docs)
#     pos_tagged_sents = nltk.pos_tag_sents(processed_docs)

#     for i in range (0, no_docs):
#     	print ner_tagged_sents[i][0],pos_tagged_sents[i][0]
