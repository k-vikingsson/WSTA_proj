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

def get_dist_to_question_word(closed_class_words, sentence_words, entity):
	# # print closed_class_words
	# answer = entity[1].lower()
	# answer_words = nltk.word_tokenize(answer)
	# # cannot proceed if answer word not found in sentence
	# for w in answer_words:
	# 	if w not in sentence_words:
	# 		return None
	# get positions of closed class question words
	question_words_pos = []
	for w in closed_class_words:
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
	return min(dists)

def make_answer_cmp_func(question, doc_set, sentences):
	"""Make comparision function of answers.

	Args:

		doc_set [str]: a list of all answers, indexed by answer id

	Returns:
		comparision funtion of two answers, more relavant answer
		is considered greater
	"""
	question_words = { w.lower() for w in word_tokenizer.tokenize(question) }
	closed_class_words = get_closed_class_words(question_words)
	question_type = get_question_type(question_words)
	def cmp_answer(a, b):
		# First, answers whose content words all appear
		# in the question should be ranked lowest.
		a_words = word_tokenizer.tokenize(a[1].lower())
		b_words = word_tokenizer.tokenize(b[1].lower())
		a_all_appear = contains_all(question_words, a_words)
		b_all_appear = contains_all(question_words, b_words)
		if a_all_appear != b_all_appear:
			return b_all_appear - a_all_appear
		# Second, answers which match the question type
		# should be ranked higher than those that don't;
		a_matches_type = a[2] == question_type
		b_matches_type = b[2] == question_type
		if a_matches_type != b_matches_type:
			return a_matches_type - b_matches_type
		# consider relavance in sentence retrieval
		a_rank = sentences.index(a[0])
		b_rank = sentences.index(b[0])
		if a_rank != b_rank:
			return b_rank - a_rank
		# Third, among entities of the same type, the
		# prefered entity should be the one which is closer
		# in the sentence to a closed-class word from the question.
		a_sent_words = [ w.lower() for w in word_tokenizer.tokenize(doc_set[a[0]]) ]
		b_sent_words = [ w.lower() for w in word_tokenizer.tokenize(doc_set[b[0]]) ]
		a_dist = get_dist_to_question_word(closed_class_words, a_sent_words, a)
		b_dist = get_dist_to_question_word(closed_class_words, b_sent_words, b)
		if a_dist != b_dist:
			if a_dist == None:
				return -1
			elif b_dist == None:
				return 1
			else:
				return b_dist - a_dist
	return cmp_answer


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
	cmp_func = make_answer_cmp_func(question, doc_set, sentences)
	key_func = cmp_to_key(cmp_func)
	return max(answers, key=key_func)
	# return get_top_answers(question, answers, doc_set, sentences)[0]

def get_top_answers(question, answers, doc_set, sentences):
	cmp_func = make_answer_cmp_func(question, doc_set, sentences)
	key_func = cmp_to_key(cmp_func)
	return sorted(answers, reverse=True, key=key_func)[:20]