from nltk.parse.stanford import StanfordParser

import os

import nltk
from nltk.tree import ParentedTree

os.environ['CLASSPATH'] = '$CLASSPATH:/usr/local/share/stanford-parser/stanford-parser.jar:/usr/local/share/stanford-parser/stanford-parser-3.7.0-models.jar'
# os.environ['STANFORD_PARSER'] = "stanford-parser.jar"
# os.environ['STANFORD_MODELS'] = "$STANFORD_MODELS:stanford-parser-3.7.0.models.jar"

parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")



# adapted from https://stackoverflow.com/questions/28681741/find-a-path-in-an-nltk-tree-tree
def get_lca_length(location1, location2):
    i = 0
    while i < len(location1) and i < len(location2) and location1[i] == location2[i]:
        i+=1
    return i

def get_labels_from_lca(ptree, lca_len, location):
    labels = []
    for i in range(lca_len, len(location)):
        labels.append(ptree[location[:i]].label())
    return labels

def findPath(ptree, text1, text2):
    leaf_values = ptree.leaves()
    leaf_index1 = leaf_values.index(text1)
    leaf_index2 = leaf_values.index(text2)

    location1 = ptree.leaf_treeposition(leaf_index1)
    location2 = ptree.leaf_treeposition(leaf_index2)

    #find length of least common ancestor (lca)
    lca_len = get_lca_length(location1, location2)

    #find path from the node1 to lca

    labels1 = get_labels_from_lca(ptree, lca_len, location1)
    #ignore the first element, because it will be counted in the second part of the path
    result = labels1[1:]
    #inverse, because we want to go from the node to least common ancestor
    result = result[::-1]

    #add path from lca to node2
    result = result + get_labels_from_lca(ptree, lca_len, location2)
    return result

# ptree = ParentedTree.fromstring("(VP (VERB saw) (NP (DET the) (NOUN dog)))")
# print(ptree.pprint())
# print(findPath(ptree, 'the', "dog"))

def cfg_path_dist(sentence, part_a, part_b):
    return cfg_path_dist_tagged(nltk.pos_tag(nltk.word_tokenize(sentence)), nltk.word_tokenize(part_a), nltk.word_tokenize(part_b))

def cfg_path_dist_tagged(sentence_tagged, a_words, b_words):
    try:
        tree = next(parser.tagged_parse(sentence_tagged))
    except ValueError:
        return None

    dists = []
    for a in a_words:
        for b in b_words:
            try:
                dists.append(len(findPath(tree, a, b)))
            except ValueError:
                pass
    return min(dists) if dists else None
    
    

# print cfg_path_dist("Microsoft was founded by Bill Gates", "founded", "Gates")
# print next(parser.tagged_parse([(u'the', 'DT'), (u'final', 'JJ'), (u'son', 'NN'), (u'of', 'IN'), (u'abd', 'NN'), (u'al', 'SYM'), (u'-', ':'), (u'malik', 'NN'), (u'to', 'TO'), (u'become', 'VB'), (u'caliph', 'NN'), (u'was', 'VBD'), (u'hisham', 'VBN'), (u'(', '('), (u'724', 'CD'), (u'\u2013', 'RB'), (u'43', 'CD'), (u'),', 'NNS'), (u'whose', 'WP$'), (u'long', 'JJ'), (u'and', 'CC'), (u'eventful', 'JJ'), (u'reign', 'NN'), (u'was', 'VBD'), (u'above', 'IN'), (u'all', 'DT'), (u'marked', 'VBN'), (u'by', 'IN'), (u'the', 'DT'), (u'curtailment', 'NN'), (u'of', 'IN'), (u'military', 'JJ'), (u'expansion', 'NN'), (u'.', '.')]))