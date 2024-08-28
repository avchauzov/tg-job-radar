import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams


nltk.download('punkt_tab')


def clean_text(text):
	text = re.sub(r'[^\w\s]', ' ', text)  # Remove all punctuation
	text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
	return text.strip().lower()


def iterate_lines(text):
	return text.split('\n')


def ngram_search(line, keywords, n=1):
	tokens = word_tokenize(line)
	ngram_list = list(ngrams(tokens, n))
	ngram_str_list = [' '.join(token) for token in ngram_list]
	intersection = list(set(ngram_str_list) & set(keywords))
	
	if len(intersection) > 0:
		return True
	
	return False
