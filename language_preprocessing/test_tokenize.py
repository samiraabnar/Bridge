from language_preprocessing.tokenize import naive_word_level_tokenizer


if __name__ == '__main__':
  test_text = 'Harry said: I am tall!'
  tokenized_test_text = naive_word_level_tokenizer(test_text)
  print(test_text, tokenized_test_text)