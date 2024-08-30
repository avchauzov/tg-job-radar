import json
import re


def load_json_into_dict(file_path):
	with open(file_path, 'r') as file:
		return json.load(file)


def replace_non_alpha_with_space(text):
	return re.sub(r'[^a-zA-Z]', ' ', text)


def remove_consecutive_spaces(text):
	return re.sub(r'\s+', ' ', text).strip()


def validate_keywords(keywords):
	return [remove_consecutive_spaces(replace_non_alpha_with_space(value)).strip().lower() for value in keywords]
