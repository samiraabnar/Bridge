import tensorflow as tf
import tensorflow_hub as hub
from util.misc import pad_lists

class TextEncoder(object):
  def __init__(self, hparams):
    self.hparams = hparams
    self.embedding_dir = self.hparams.embedding_dir

  def get_embeddings(self, text, sequences_length):
    raise NotImplementedError()

  def get_embeddings_values(self, text_sequences, key='elmo'):
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.table_initializer())
      print(text_sequences)
      embeddings = sess.run(self.get_embeddings(text_sequences, key))
    return embeddings

  def save_embeddings(self, text):
    raise NotImplementedError()

  def load_saved_embeddings(self):
    if self.embedding_dir is not None:
      print('loading embeddings')



class TfHubElmoEncoder(TextEncoder):
  """ELMO (tf.hub module).
   """
  def __init__(self, hparams, trainable=False):
    super(TfHubElmoEncoder, self).__init__(hparams)
    self.embedder = hub.Module('https://tfhub.dev/google/elmo/2', trainable=trainable)

  def get_embeddings(self, text_sequences, sequences_length, key='elmo'):
    """

    :param text_sequences:
    :param sequences_length:
    :param key:
            word_emb: the character-based word representations with shape [batch_size, max_length, 512].
            lstm_outputs1: the first LSTM hidden state with shape [batch_size, max_length, 1024].
            lstm_outputs2: the second LSTM hidden state with shape [batch_size, max_length, 1024].
            elmo: the weighted sum of the 3 layers, where the weights are trainable. This tensor has shape [batch_size, max_length, 1024]
            default: a fixed mean-pooling of all contextualized word representations with shape [batch_size, 1024].
    :return:
    """
    padded_text_sequences = pad_lists(text_sequences)
    embeddings = self.embedder(
      inputs={'tokens': padded_text_sequences,
              'sequence_len': sequences_length }
      ,signature='tokens', as_dict=True)[key]

    return embeddings

  def get_text_embedding_column(key='elmo'):
    return hub.text_embedding_column(
      key=key,
      module_spec="https://tfhub.dev/google/elmo/2")


class TfHubUniversalEncoder(TextEncoder):
  """Google Universal Sentence Encoder (tf.hub module).
  """
  def __init__(self, hparams, trainable=False):
    super(TfHubUniversalEncoder, self).__init__(hparams)
    self.embedder = hub.Module('https://tfhub.dev/google/universal-sentence-encoder/2', trainable=trainable)

  def get_embeddings(self, text_sequences, key='sentence'):
    print("text:", text_sequences)
    embeddings = self.embedder(
      text_sequences,
      signature="default",
      as_dict=True)[key]

    return embeddings


  def get_text_embedding_column(key='sentence'):
    return hub.text_embedding_column(
      key="sentence",
      module_spec="https://tfhub.dev/google/universal-sentence-encoder/2")


class BertEncoder(TextEncoder):
  def __init__(self):
    raise NotImplementedError()
