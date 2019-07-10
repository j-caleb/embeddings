import commons
from collections import defaultdict

stopwords = commons.get_data('../data/stoplist.txt')
stopwords = [el.strip() for el in stopwords]
stopwords = set(stopwords)

def strip_stopwords(line):
	line = line.split()
	line = [token for token in line if token not in stopwords]
	return ' '.join(line)

def strip_nonvalid_tokens(line, valid_tokens):
	"""
	This strips stopwords and also any tokens that are not valid.
	"""
	line=line.split()
	line=[token for token in line if token not in stopwords and token in valid_tokens]
	return ' '.join(line)

def get_valid_terms(data, min_count):
	counts = get_term_counts(data)
	remove = [token for token in counts if counts[token] <= min_count]
	for token in remove:
		del counts[token]
	return list(counts.keys())

def get_term_counts(data):
	counts = defaultdict(float)
	for line in data:
		line = strip_stop_words(line).split()
		for token in line:
			counts[token]+=1
	return counts

def create_context_training(line, widow_size, valid_tokens):
	"""
	This is used to select terms based on a context window. The terms are selected
	before and after the target term with the width of the window being determined
	by window_size.
	"""
    contexts = []
    line = strip_nonvalid_tokens(line).split()
    for i in range(len(line)):
        target = line[i]
        start = 0
        end = len(line)
        if i-window_size > 0:
            start=i-window_size
        if i+window_size+1<len(line):
            end=i+window_size+1
        before = line[start:i]
        after = line[i+1:end]
        contexts.append([target,before + after])
    return contexts

# def get_document_term_frequency():
# 	"""
# 	For some statistics both document and term frequency are needed. This generates
# 	both in one pass.
# 	"""
# 	return True
