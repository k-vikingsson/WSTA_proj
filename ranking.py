import nltk
word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()

common_measurements = set()
with open("common_measurements.txt") as file:
	for line in file:
		common_measurements.add(line.strip())

common_localities = set()
with open("common_localities.txt") as file:
	for line in file:
		common_localities.add(line.strip())

def get_question_type(question_words):
	"""Determine question type.

	Args:
		question (str): the question as a string

	Returns:
		(str): type of question as a string
	"""
	# TODO more rules
	if "who" in question_words:
		return "PERSON"
	elif "where" in question_words:
		return "LOCATION"
	elif "how" in question_words and "many" in question_words:
		return "NUMBER"
	elif "what" in question_words and "year" in question_words:
		return "NUMBER"
	elif "when" in question_words:
		return "NUMBER"
	else:
		if "what" in question_words or "which" in question_words:
			if "king" in question_words: return "PERSON"
			elif "name" in question_words: return "PERSON"
			for w in question_words:
				if w in common_measurements: return "NUMBER"
				elif w in common_localities: return "LOCATION"
		return "OTHER"

def contains_all(items, elems):
	for e in elems:
		if e not in items:
			return False
	return True

# def get_score(question, answer, doc_set, sentences):
# 	"""Calculate the score of an answer given a question.

# 	Args:
# 		question (str): the question as a string
# 		answers [(str, str, str)]: an answer to the question
# 			being a 3-tuple of (sentence, entity, entity type)
# 		doc_set [str]: a list of all answers, indexed by answer id

# 	Returns:
# 		(float): score of answer
# 	"""
# 	# First, answers whose content words all appear
# 	# in the question should be ranked lowest.
# 	ans_words = word_tokenizer.tokenize(doc_set[answer[0]])
# 	question_words = set(word_tokenizer.tokenize(question))
# 	if contains_all(question_words, ans_words):
# 		return 0.0

# 	# Second, answers which match the question type
# 	# should be ranked higher than those that don't;
# 	if answer[2] != get_question_type(question):
# 		return 1.0

# 	# Third, among entities of the same type, the
# 	# prefered entity should be the one which is closer
# 	# in the sentence to a closed-class word from the question.
# 	# TODO

# 	rank = sentences.index(answer[0])
# 	return 1.0 + 100.0/(rank+1)

def get_closed_class_words(question_words):
	tagged = nltk.pos_tag(question_words, tagset="universal")
	# consider pronouns, determiners, conjunctions, and prepositions as closed class
	return [p[0] for p in tagged if p[1] in ["PRON", "DET", "CONJ", "ADP", "AUX", "NUM", "PART"]]

def get_open_class_words(question_words):
	tagged = nltk.pos_tag(question_words, tagset="universal")
	# consider pronouns, determiners, conjunctions, and prepositions as closed class
	return [p[0] for p in tagged if p[1] in ["ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"]]

def get_dist_to_question_word(target_words, sentence_words, entity):
	# # print closed_class_words
	# answer = entity[1].lower()
	# answer_words = nltk.word_tokenize(answer)
	# # cannot proceed if answer word not found in sentence
	# for w in answer_words:
	# 	if w not in sentence_words:
	# 		return None
	# get positions of closed class question words
	question_words_pos = []
	for w in target_words:
		for i in range(len(sentence_words)):
			if w == sentence_words[i]:
				question_words_pos.append(i)
	# print question_words_pos
	# cannot proceed if no such closed word in sentence
	if len(question_words_pos) == 0:
		return None
	# # (naive way to) find answer position in sentence
	# answer_start_pos = sentence_words.index(answer_words[0])
	# answer_end_pos = sentence_words.index(answer_words[-1])
	answer_start_pos = entity[3]
	answer_end_pos = entity[4]
	# print "ans start", answer_start_pos
	# print "ans end", answer_end_pos
	# calculate distance and find closest
	dists = [ min(abs(p-answer_start_pos), abs(p-answer_end_pos))
		for p in question_words_pos ]
	# print dists
	return sum(dists)

def cmp_answer(a, b):
	# First, answers whose content words all appear
	# in the question should be ranked lowest.
	a_all_appear = a[-3]
	b_all_appear = b[-3]
	if a_all_appear != b_all_appear:
		return b_all_appear - a_all_appear
	# Second, answers which match the question type
	# should be ranked higher than those that don't;
	a_matches_type = a[-2]
	b_matches_type = b[-2]
	if a_matches_type != b_matches_type:
		return a_matches_type - b_matches_type
	# consider relavance (rank) in sentence retrieval
	a_rank = a[-4]
	b_rank = b[-4]
	if a_rank != b_rank:
		return b_rank - a_rank
	# Third, among entities of the same type, the
	# prefered entity should be the one which is closer
	# in the sentence to a closed-class word from the question.
	a_dist = a[-1]
	b_dist = b[-1]
	if a_dist != b_dist:
		if a_dist == None:
			return -1
		elif b_dist == None:
			return 1
		else:
			return b_dist - a_dist
	return 0

def add_answer_properties(question_words, question_type, open_class_words, answer, doc_set, sentences):
	answer_words = word_tokenizer.tokenize(answer[1].lower())
	answer_sent_words = [ w.lower() for w in word_tokenizer.tokenize(doc_set[answer[0]]) ]
	added = list(answer)
	added.append(sentences.index(answer[0]))
	added.append(contains_all(question_words, answer_words))
	added.append(answer[2] == question_type)
	added.append(get_dist_to_question_word(open_class_words, answer_sent_words, answer))
	return tuple(added)

from functools import cmp_to_key
def get_best_answer(question, answers, doc_set, sentences):
	"""Return the best answer from answers to a question.

	Args:
		answers [(str, str, str)]: a list of answers,
			each being a 3-tuple of (sentence, entity, entity type)
		doc_set [str]: a list of all answers, indexed by answer id
		sentences [int]: a list of sentence id's sorted by relavance

	Returns:
		(str, str, str): the best answer to the question
	"""
	question_words = { w.lower() for w in word_tokenizer.tokenize(question) }
	question_type = get_question_type(question_words)
	open_class_words = get_open_class_words(question_words)
	answers_added = [
		add_answer_properties(
			question_words,
			question_type,
			open_class_words,
			a,
			doc_set,
			sentences)
		for a in answers
	]
	key_func = cmp_to_key(cmp_answer)
	return max(answers_added, key=key_func)
	# return get_top_answers(question, answers, doc_set, sentences)[0]

def get_top_answers(question, answers, doc_set, sentences):
	question_words = { w.lower() for w in word_tokenizer.tokenize(question) }
	question_type = get_question_type(question_words)
	open_class_words = get_open_class_words(question_words)
	answers_added = [
		add_answer_properties(
			question_words,
			question_type,
			open_class_words,
			a,
			doc_set,
			sentences)
		for a in answers
	]
	key_func = cmp_to_key(cmp_answer)
	return sorted(answers_added, reverse=True, key=key_func)[:20]