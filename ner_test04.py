import os
import nltk
import json
from nltk.tag import StanfordNERTagger

# nltk.download('maxent_treebank_pos_tagger')
# nltk.download('averaged_perceptron_tagger')

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
    for token, tag, i in tagged_sent:
        if tag != cur_tag:
            pos = tagged_sent.index((token,tag,i))
            return pos

def get_continuous_chunks(tagged_sents):
    sents_chunks = []
    for tagged_sent in tagged_sents:
        tagged_sent.append(('end','END','END'))
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

def get_tagged(processed_docs,no_docs):
    ner_tagged_sents = st.tag_sents(processed_docs)
    pos_tagged_sents = nltk.pos_tag_sents(processed_docs)
    tagged_sents = []

    for j in range (0,no_docs):
        tagged_sent =[]
        no_tokens = len(ner_tagged_sents[j])

        for i in range (0,no_tokens):
            token = ner_tagged_sents[j][i][0]
            tag = ner_tagged_sents[j][i][1]


            if token != '':
                if tag != 'O':
                    tagged_sent.append((token,tag,i))

                else:
                    if i != 0 and token[0].isupper():
                        tag = 'OTHER'
                        tagged_sent.append((token,tag,i))

                    elif pos_tagged_sents[j][i][1] == 'CD':
                        # 5/five people / meters
                        if no_tokens > i+1:
                            if pos_tagged_sents[j][i+1][1] == 'NN' or pos_tagged_sents[j][i+1][1] == 'NNS':
                                token = token + ' ' + pos_tagged_sents[j][i+1][0]

                            # 14,372
                            elif (pos_tagged_sents[j][i+1][1] == ',' or pos_tagged_sents[j][i+1][1] == '.') and no_tokens > i+2 and pos_tagged_sents[j][i+2][1] == 'CD':
                                token = token + pos_tagged_sents[j][i+1][0] + pos_tagged_sents[j][i+2][0]
                                tem = pos_tagged_sents[j].pop(i+2)
                                tem_token = tem[0]
                                pos_tagged_sents[j].insert(i+2,(tem_token,'NONE'))

                                # 16,290 people
                                if no_tokens > i + 3 and (pos_tagged_sents[j][i+3][1] == 'NN' or pos_tagged_sents[j][i+3][1] == 'NNS'):
                                    token = token + ' ' + pos_tagged_sents[j][i+3][0]


                                # 1.5 million
                                elif no_tokens > i + 3 and pos_tagged_sents[j][i+3][1] == 'CD':
                                    token = token + ' ' + pos_tagged_sents[j][i+3][0]
                                    tem1 = pos_tagged_sents[j].pop(i+3)
                                    tem1_token = tem1[0]
                                    pos_tagged_sents[j].insert(i+2,(tem1_token,'NONE'))

                                    if no_tokens > i + 4 and (pos_tagged_sents[j][i+4][1] == 'NN' or pos_tagged_sents[j][i+4][1] == 'NNS'):
                                        token = token + ' ' + pos_tagged_sents[j][i+4][0]





                            tag = 'NUMBER'
                            tagged_sent.append((token,tag,i))


                            pos_tagged_sents[j][i+1][1] == ','


                    else:
                        tagged_sent.append((token,tag,i))

            else:
                tagged_sent.append((token,tag,i))

        tagged_sents.append(tagged_sent)
    return tagged_sents




def parse_docs(doc_set):
    processed_docs = process_doc_ner(doc_set)
    no_docs = len(processed_docs)
    tagged_sents = get_tagged(processed_docs,no_docs)

    name_entity_list = get_continuous_chunks(tagged_sents)

    doc_ne_pairs = []
    for i in range (0,no_docs):
        name_entity = name_entity_list[i]
        # name_entity_str = [" ".join([token for token, tag in ne]) for ne in name_entity]
        name_entity_pairs = [(i," ".join([token for token, tag, start in ne]), ne[0][1],ne[0][2],ne[-1][2]) for ne in name_entity]
        for sent_id,entity,tag,start_i,end_i in name_entity_pairs:
            if tag != 'O':
                if tag == 'ORGANIZATION':
                    doc_ne_pairs.append({'id':sent_id,'answer':entity,'type':'OTHER','start_pos':start_i,'end_pos':end_i})
                elif tag == 'NUMBER':
                    text = word_tokenizer.tokenize(entity)
                    n = len(text)
                    if n != 1:
                        end_i = end_i + n - 1
                    doc_ne_pairs.append({'id':sent_id,'answer':entity,'type':tag,'start_pos':start_i,'end_pos':end_i})
                else:
                    doc_ne_pairs.append({'id':sent_id,'answer':entity,'type':tag,'start_pos':start_i,'end_pos':end_i})
    return doc_ne_pairs

if __name__ == '__main__':
    with open('QA_dev.json') as dev_file:
        dev = json.load(dev_file)

    for i in range (0,5):
        doc_set = [s['answer'] for s in dev[i]['qa']]
        for i in range(len(doc_set)):
            answers = parse_docs([doc_set[i]])
            print answers
            print doc_set[i]
            print ''

