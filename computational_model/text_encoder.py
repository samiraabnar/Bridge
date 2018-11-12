import tensorflow as tf
import tensorflow_hub as hub


class TextEncoder(object):
  def __init__(self, hparams):
    self.hparams = hparams
    self.embedding_dir = self.hparams.embedding_dir

  def get_embeddings(self, text):
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

  def get_embeddings(self, text_sequences, key='elmo'):
    embeddings = self.embedder(
      text_sequences,
      signature="default",
      as_dict=True)[key]

    return embeddings

  def get_text_embedding_column(key='elmo'):
    return hub.text_embedding_column(
      key="sentence",
      module_spec="https://tfhub.dev/google/elmo/2")


class TfHubUniversalEncoder(TextEncoder):
  """Google Universal Sentence Encoder (tf.hub module).
  """
  def __init__(self, hparams, trainable=False):
    super(TfHubUniversalEncoder, self).__init__(hparams)
    self.embedder = hub.Module('https://tfhub.dev/google/universal-sentence-encoder/2', trainable=trainable)

  def get_embeddings(self, text_sequences, key='elmo'):
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
