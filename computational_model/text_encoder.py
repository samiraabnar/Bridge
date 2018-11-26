import tensorflow as tf
import tensorflow_hub as hub
from util.misc import pad_lists, concat_lists
from util.word_embedings import WordEmbeddingLoader
import numpy as np
from google_lm.interface_util import GoogleLMInterface
from tqdm import tqdm

class TextEncoder(object):
  def __init__(self, hparams):
    self.hparams = hparams
    self.embedding_dir = self.hparams.embedding_dir

  def get_embeddings(self, text, sequences_length):
    raise NotImplementedError()

  def get_embeddings_values(self, text_sequences, sequences_length, key='elmo'):
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      embeddings = sess.run(self.get_embeddings(text_sequences,sequences_length, key))
    return embeddings

  def save_embeddings(self, text):
    raise NotImplementedError()

  def load_saved_embeddings(self):
    if self.embedding_dir is not None:
      print('loading embeddings')



class GoogleLMEncoder(TextEncoder):
  def __init__(self, hparams, path):
    super(GoogleLMEncoder, self).__init__(hparams)
    self.path = path

  def get_embeddings_values(self, text_sequences, sequences_length, key='lstm_0'):
    self.google_lm_interface = GoogleLMInterface(self.path)

    embeddings = {'lstm_0': [], 'lstm_1': []}

    if self.hparams.context_mode == 'block':
      concatenated_text_sequences = concat_lists(text_sequences)
      whole_block = ' '.join(concatenated_text_sequences)
      with self.google_lm_interface.sess as sess:
        self.google_lm_interface.sess.run(self.google_lm_interface.tensors['states_init'])
        lstm_1, lstm_2 = self.google_lm_interface.forward(sess, whole_block)
      for text_sequence in tqdm(text_sequences):
        embeddings['lstm_0'].append(lstm_1)
        embeddings['lstm_1'].append(lstm_2)
    else:
      embeddings_dic = {}
      with self.google_lm_interface.sess as sess:
        for text_sequence in tqdm(text_sequences):
          self.google_lm_interface.sess.run(self.google_lm_interface.tensors['states_init'])
          current_input = ' '.join(text_sequence)
          if current_input not in embeddings_dic:
            lstm_1, lstm_2 = self.google_lm_interface.forward(sess, current_input)
            embeddings_dic[current_input] = (lstm_1, lstm_2)

          embeddings['lstm_0'].append(embeddings_dic[current_input][0])
          embeddings['lstm_1'].append(embeddings_dic[current_input][1])

    return np.asarray(embeddings[key])


class GloVeEncoder(TextEncoder):
  def __init__(self, hparams, dataset, dim=300):
    super(GloVeEncoder, self).__init__(hparams)
    self.dim = dim
    self.we = WordEmbeddingLoader(self.hparams.embedding_dir, dim=300)
    self.we.build_vocab(dataset)
    self.we.build_lookups()

    # Build Graph
    load_embedding_matrix = self.we.matrix['Embedding_matrix']
    self.sentences = tf.placeholder(tf.int32,
                               shape=[None, None]
                               )
    self.Word_embedding = tf.get_variable(name="Word_embedding",
                                     shape=[len(self.we.vocabulary), self.dim],
                                     initializer=tf.constant_initializer(np.array(load_embedding_matrix)),
                                     trainable=False
                                     )


  def get_embeddings(self, text_sequences, text_sequences_length=None, key=None):
    """

    :param text_sequences_length: list of strings.
    :return:
    """
    concatenated_text_sequences = concat_lists(text_sequences)

    encoded_inputs = self.we.encode(concatenated_text_sequences)
    embeddings = tf.nn.embedding_lookup(self.Word_embedding, tf.cast(encoded_inputs, dtype=tf.int32))

    return embeddings

class TfTokenEncoder(TextEncoder):
  def __init__(self, hparams):
    super(TfTokenEncoder, self).__init__(hparams)
    self.embedder = hub.Module("https://tfhub.dev/google/nnlm-en-dim128/1")

  def get_embeddings(self, text_sequences, text_sequences_length=None, key=None):
    """

    :param text_sequences_length: list of list.
    :return:
      embeddings [batch_size, embedding_size]--> one embedding per each list
    """
    concatenated_text_sequences = concat_lists(text_sequences)
    print(len(concatenated_text_sequences))
    embeddings = self.embedder(concatenated_text_sequences)

    print(embeddings.shape)
    return embeddings

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

  def get_embeddings(self, text_sequences, text_sequences_length=None, key='sentence'):
    concatenated_text_sequences = concat_lists(text_sequences)
    embeddings = self.embedder(
      concatenated_text_sequences)

    return embeddings

  def get_embeddings_values(self, text_sequences, sequences_length, key='elmo'):
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      embeddings = sess.run(self.get_embeddings(text_sequences,sequences_length, key))
    return embeddings


class TfHubLargeUniversalEncoder(TfHubUniversalEncoder):
  """Google Universal Sentence Encoder Large (tf.hub module).
  """

  def __init__(self, hparams, trainable=False):
    super(TfHubLargeUniversalEncoder, self).__init__(hparams)
    self.embedder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")


  def get_embeddings(self, text_sequences, text_sequences_length=None, key='sentence'):
    concatenated_text_sequences = concat_lists(text_sequences)
    embeddings = self.embedder(
      concatenated_text_sequences)

    return embeddings


  def get_text_embedding_column(key='sentence'):
    return hub.text_embedding_column(
      key="sentence",
      module_spec="https://tfhub.dev/google/universal-sentence-encoder/2")


class BertEncoder(TextEncoder):
  def __init__(self):
    raise NotImplementedError()




if __name__ == '__main__':
  FLAGS = tf.flags.FLAGS
  tf.flags.DEFINE_string('embedding_dir', None, 'path to the file containing the embeddings')
  hparams = FLAGS
  embedder = TfHubUniversalEncoder(hparams)
  with tf.Session() as sess:
    encoded_stimuli_of_each_block = {}
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    print("get the embeddings ...")
    output = sess.run(embedder.get_embeddings(['I am happy .']))
    print(output.shape)