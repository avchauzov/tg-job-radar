import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams


nltk.download('punkt_tab')


def clean_text(text):
	# Remove URLs
	text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
	# Remove all punctuation except newlines
	text = re.sub(r'[^\w\s\n]', ' ', text)
	# Replace multiple spaces with a single space (excluding newlines)
	text = re.sub(r'[^\S\n]+', ' ', text)
	return text.strip().lower()


def iterate_lines(text):
	return text.split('\n')


def ngram_search(line, keywords, n=1):
	tokens = word_tokenize(line.strip())
	ngram_list = [' '.join(ngram) for ngram in list(ngrams(tokens, n))]
	intersection = set(ngram_list).intersection(keywords)
	
	return len(intersection) > 0
