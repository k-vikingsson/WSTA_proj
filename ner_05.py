import os
import nltk
import json
from nltk.tag import StanfordNERTagger
from nltk.corpus import stopwords

# nltk.download('maxent_treebank_pos_tagger')
# nltk.download('averaged_perceptron_tagger')

os.environ['CLASSPATH'] = '/usr/local/share/stanford-ner/stanford-ner.jar'
os.environ['STANFORD_MODELS'] = '/usr/local/share/stanford-ner/classifiers/english.muc.7class.distsim.crf.ser.gz'



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
                    tagged_sent.append((token,tag,j))

                else:
                    if j == 0:
                        if token.lower() not in STOPWORDS:
                        #token[0].isupper() and token.lower() not in STOPWORDS:
                            if token in names:
                                #print 'name',token,pos_sent[j][1]

                                tag = 'PERSON'
                                tagged_sent.append((token,tag,j))



                            elif pos_sent[j][1] in ['NNP', 'NNPS']:
                                #print 'nnp',token,pos_sent[j][1]

                                tag = 'OTHER'
                                tagged_sent.append((token,tag,j))

                    elif token[0].isupper():
                        tag = 'OTHER'
                        tagged_sent.append((token,tag,j))

                        
                    elif pos_sent[j][1] == 'CD':
                        #print 'num',token,pos_sent[j][1]

                        tag = 'NUMBER'
                        tagged_sent.append((token,tag,j))

                    else:
                        tagged_sent.append((token,tag,j))

        tagged_sents.append(tagged_sent)

    return tagged_sents



import pprint
pp = pprint.PrettyPrinter(indent=4)

def parse_docs(doc_set):
    processed_docs,processed_docs_pu = process_doc_ner(doc_set)
    no_docs = len(processed_docs_pu)

    # change
    tagged_sents = ner_tagger(processed_docs_pu,no_docs)

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
    # with open('QA_dev.json') as dev_file:
    #     dev = json.load(dev_file)
    doc_set = ['They are members of the National Collegiate Athletic Association and participate in the University Athletic Association at the Division III level.','Some columnists in The Times are connected to the Conservative Party such as Daniel Finkelstein, Tim Montgomerie, Matthew Parris and Matt Ridley, but there are also columnists connected to the Labour Party such as David Aaronovitch, Phil Collins, Oliver Kamm and Jenni Russell.']  

        #doc_set = dev[i]['sentences']
    pp.pprint(parse_docs(doc_set))
    print ''