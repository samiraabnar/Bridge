import numpy as np


class WordEmbeddingLoader(object):
  def __init__(self, embeding_dir, dim):
    self.symbols = {0: 'PAD', 1: 'UNK'}
    self.vocabulary = []
    self.embedding_dir = embeding_dir
    self.dim = dim

  def build_vocab(self, dataset):
    vocab = []

    for sentence in dataset:
      for word in sentence.split():
        vocab.append(word.lower())

    vocab = list(set(vocab))

    self.vocabulary = vocab

    return vocab

  def word_embedding_matrix(self):

    # first and second vector are pad and unk words
    with open(self.embedding_dir, 'r') as f:
      word_vocab = []
      embedding_matrix = []
      word_vocab.extend(['PAD', 'UNK'])
      embedding_matrix.append(np.random.uniform(-1.0, 1.0, (1, self.dim))[0])
      embedding_matrix.append(np.random.uniform(-1.0, 1.0, (1, self.dim))[0])

      for line in f:
        if line.split()[0] in self.vocabulary:
          word_vocab.append(line.split()[0])
          embedding_matrix.append([float(i) for i in line.split()[1:]])

    return {'word_vocab': word_vocab, 'Embedding_matrix': np.reshape(embedding_matrix, [-1, self.dim]).astype(np.float32)}

  def build_lookups(self):
    self.matrix = self.word_embedding_matrix()
    shape_word_vocab = self.matrix['word_vocab']

    self.int_to_vocab = {}

    for index_no, word in enumerate(shape_word_vocab):
      self.int_to_vocab[int(index_no)] = word

    self.int_to_vocab.update(self.symbols)

    self.vocab_to_int = {word: int(index_no) for index_no, word in self.int_to_vocab.items()}

  def encode(self,inputs):
    max_length = max([len(inp) for inp in inputs])
    print("max length:", max_length)
    encoded_data = []
    for sentence in inputs:
      sentence_ = []
      for word in sentence.split():
        if word.lower() in self.vocab_to_int:
          sentence_.append(self.vocab_to_int[word.lower()])
      sentence_.extend([self.vocab_to_int['PAD']] * max([0, max_length - len(sentence_)]))
      encoded_data.append(sentence_)

    return encoded_data



if __name__ == '__main__':
  dataset = ['I like going to the cinema.',
             'Mum is going to cook spinach tonight.',
             'Do you think they are going to win the match ?',
             'Are you going to the United States ?',
             "He doesn't like playing video games."
             ]

  we = WordEmbeddingLoader('/Users/samiraabnar/Codes/Data/word_embeddings/glove.6B/glove.6B.100d.txt')
  we.build_vocab(dataset)
  we.build_lookups()



  print(we.vocab_to_int)

  encoded_data = we.encode(dataset)

  # check
  for i in encoded_data:
    print([we.int_to_vocab[j] for j in i])