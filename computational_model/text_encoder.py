import sys
sys.path.append("/Users/samiraabnar/Codes/bert")
sys.path.append("/Users/samiraabnar/Codes/")

import tensorflow as tf
import tensorflow_hub as hub
from util.misc import pad_lists, concat_lists
from util.word_embedings import WordEmbeddingLoader
import numpy as np
from google_lm.interface_util import GoogleLMInterface
from tqdm import tqdm
import codecs
import collections
import json
import re
import bert
import bert.modeling
import bert.tokenization
import bert.extract_features_utils as extract_features

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
    return np.asarray(embeddings)

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
    print('Glove Embeddings')
    concatenated_text_sequences = concat_lists(text_sequences)
    print('concatenated_text_sequences')
    encoded_inputs = self.we.encode(concatenated_text_sequences)
    print(len(encoded_inputs))
    embeddings = tf.nn.embedding_lookup(self.Word_embedding, tf.cast(encoded_inputs, dtype=tf.int32))

    return embeddings

class TfTokenEncoder(TextEncoder):
  def __init__(self, hparams):
    super(TfTokenEncoder, self).__init__(hparams)
    self.embedder = hub.Module("https://tfhub.dev/google/nnlm-en-dim128/1")

  def get_embeddings_values(self, text_sequences, sequences_length, key='elmo'):
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      embeddings = sess.run(self.get_embeddings(text_sequences,sequences_length, key))
    return embeddings

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
      module_spec="https://tfhub.dev/google/universal-sentence-encoder-large/3")


class BertEncoder(TextEncoder):
  def __init__(self, hparams):
    self.vocab_file = hparams.root+"/bert/pretrained_models/cased_L-12_H-768_A-12/vocab.txt"
    self.config_file = hparams.root+"/bert/pretrained_models/cased_L-12_H-768_A-12/bert_config.json"
    self.init_checkpoint = hparams.root+"/bert/pretrained_models/cased_L-12_H-768_A-12/bert_model.ckpt"

    self.layer_indexes = [int(x) for x in hparams.layers.split(",")]

    self.bert_config = bert.modeling.BertConfig.from_json_file(self.config_file)

    self.tokenizer = bert.tokenization.FullTokenizer(
      vocab_file=self.vocab_file, do_lower_case=False)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    self.run_config = tf.contrib.tpu.RunConfig(
      master=None,
      tpu_config=tf.contrib.tpu.TPUConfig(
        num_shards=None,
        per_host_input_for_training=is_per_host))


  def convert_example(self, input_sequences):
    examples = []
    unique_id = 0
    for sequence in input_sequences:
      line = bert.tokenization.convert_to_unicode(sequence)
      line = line.strip()
      text_a = None
      text_b = None
      m = re.match(r"^(.*) \|\|\| (.*)$", line)
      if m is None:
        text_a = line
      else:
        text_a = m.group(1)
        text_b = m.group(2)
      examples.append(
        extract_features.InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
      unique_id += 1

    return examples

  def get_embeddings_values(self, text_sequences, sequences_length, key=None):
    examples = self.convert_example(text_sequences)

    features = extract_features.convert_examples_to_features(
      examples=examples, seq_length=128, tokenizer=self.tokenizer)

    unique_id_to_feature = {}
    for feature in features:
      unique_id_to_feature[feature.unique_id] = feature

    model_fn = extract_features.model_fn_builder(
      bert_config=self.bert_config,
      init_checkpoint=self.init_checkpoint,
      layer_indexes=self.layer_indexes,
      use_tpu=False,
      use_one_hot_embeddings=False)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=False,
      model_fn=model_fn,
      config=self.run_config,
      predict_batch_size=32)

    input_fn = extract_features.input_fn_builder(
      features=features, seq_length=128)

    output_embeddings = []
    for result in estimator.predict(input_fn, yield_single_examples=True):
      unique_id = int(result["unique_id"])
      feature = unique_id_to_feature[unique_id]

      all_features = []
      for (i, token) in enumerate(feature.tokens):
        all_layers = []
        for (j, layer_index) in enumerate(self.layer_indexes):
          layer_output = result["layer_output_%d" % j]
          layers = collections.OrderedDict()
          layers["index"] = layer_index
          layers["values"] = [
            round(float(x), 6) for x in layer_output[i:(i + 1)].flat
          ]

          all_layers.append(layers)
        features = collections.OrderedDict()
        features["token"] = token
        features["layers"] = all_layers
        all_features.append(features)
      output_embeddings.append(all_features)
    return output_embeddings

import os

def test():
  FLAGS = tf.flags.FLAGS
  tf.flags.DEFINE_integer('subject_id', 1, 'subject id')
  tf.flags.DEFINE_integer('fold_id', 0, 'fold id')
  tf.flags.DEFINE_list('delays', [-6, -4, -2, 0], 'delay list')
  tf.flags.DEFINE_boolean('cross_delay', False, 'try different train and test delays')

  tf.flags.DEFINE_float('alpha', 1, 'alpha')
  tf.flags.DEFINE_string('embedding_dir', 'Data/word_embeddings/glove.6B/glove.6B.300d.txt',
                         'path to the file containing the embeddings')
  tf.flags.DEFINE_string('brain_data_dir', 'Data/harrypotter/', 'Brain Data Dir')
  tf.flags.DEFINE_string('root', '/Users/samiraabnar/Codes/', 'general path root')

  tf.flags.DEFINE_enum('text_encoder', 'universal_large',
                       ['glove', 'elmo', 'tf_token', 'universal_large', 'google_lm'], 'which encoder to use')
  tf.flags.DEFINE_string('embedding_type', '', 'ELMO: word_emb, lstm_outputs1, lstm_outputs2, elmo ')
  tf.flags.DEFINE_string('context_mode', 'none', 'type of context (sentence, block, none)')
  tf.flags.DEFINE_integer('past_window', 3, 'window size to the past')
  tf.flags.DEFINE_integer('future_window', 0, 'window size to the future')
  tf.flags.DEFINE_boolean('only_past', True, 'window size to the future')

  tf.flags.DEFINE_boolean('save_data', True, 'save data flag')
  tf.flags.DEFINE_boolean('load_data', True, 'load data flag')
  tf.flags.DEFINE_boolean('save_encoded_stimuli', True, 'save encoded stimuli')
  tf.flags.DEFINE_boolean('load_encoded_stimuli', True, 'load encoded stimuli')

  tf.flags.DEFINE_boolean('save_models', True, 'save models flag')

  tf.flags.DEFINE_string("param_set", None, "which param set to use")
  tf.flags.DEFINE_string("emb_save_dir", 'bridge_models/embeddings/', 'where to save embeddings')

  hparams = FLAGS

  hparams.embedding_dir = os.path.join(hparams.root, hparams.embedding_dir)
  hparams.brain_data_dir = os.path.join(hparams.root, hparams.brain_data_dir)
  harrypotter_clean_sentences = np.load(os.path.join(hparams.brain_data_dir,"harrypotter_cleaned_sentences.npy"))
  saving_dir = os.path.join(hparams.root, hparams.emb_save_dir,hparams.text_encoder +"_"+ hparams.embedding_type +"_"+ hparams.context_mode
                            + "_" + str(hparams.past_window) +"-"+ str(hparams.future_window) + "_onlypast-" + str(hparams.only_past))
  print("brain data dir: ", hparams.brain_data_dir)
  print("saving dir: ", saving_dir)


  TextEncoderDic = {'elmo':TfHubElmoEncoder(hparams),
                    'tf_token': TfTokenEncoder(hparams),
                    'universal_large': TfHubLargeUniversalEncoder(hparams),
                    'glove': GloVeEncoder(hparams, harrypotter_clean_sentences),
                    'google_lm': GoogleLMEncoder(hparams, path= os.path.join(hparams.root,"GoogleLM1b/"))
                    }

  indexes = [1,2]
  glove_embedding = TextEncoderDic['glove'].get_embeddings_values([['this','is','very','crazy'], ['what', 'is', 'your', 'name','?']],None,None)
  uni_embedding = TextEncoderDic['google_lm'].get_embeddings_values([['this','is','very','crazy'], ['what', 'is', 'your', 'name' '?']],None,'lstm_0')


  def integration_fn(inputs, axis, max_size=200000):
    if len(inputs.shape) > 1:
      inputs = np.mean(inputs, axis=axis)
    size = inputs.shape[-1]
    return inputs[:np.min([max_size, size])]

  print(uni_embedding.shape)
  if len(glove_embedding.shape) > 2:
    glove_embedding = integration_fn(glove_embedding[0][indexes], axis=0)
  if len(uni_embedding.shape) > 2:
    uni_embedding = integration_fn(uni_embedding[0][indexes], axis=0)
  print(glove_embedding.shape)
  print(uni_embedding.shape)

if __name__ == '__main__':
  FLAGS = tf.flags.FLAGS
  tf.flags.DEFINE_string('layers', '0,1,2,3,4,5,6,7,8,9,10,11', 'layers')
  tf.flags.DEFINE_string('root', '/Users/samiraabnar/Codes', 'layers')

  hparams = FLAGS

  bert_encoder_obj = BertEncoder(hparams)

  embeddings_for_each_layer={0:[],1:[],2:[],3:[],4:[],5:[],6:[],
                             7:[], 8:[], 9:[],10:[],11:[]}
  sentences = ['this is a cat']
  output_embeddings = bert_encoder_obj.get_embeddings_values(['this is a  grubby cat'],[4])



  for sent, output in zip(sentences,output_embeddings):
    print(sent)
    sent_embeddings_for_each_layer = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],
                                      7:[], 8:[], 9:[],10:[],11:[]}
    current_outputs = []
    real_ind = 0
    while real_ind < len(output):
      f = output[real_ind]
      if f['token'] != '[CLS]' and f['token'] != '[SEP]':
        if f['token'].startswith("##"):
          print("cont",f['token'])
          for layer_ind in np.arange(12):
            sent_embeddings_for_each_layer[layer_ind][-1] = np.mean([sent_embeddings_for_each_layer[layer_ind][-1],np.asarray(f['layers'][layer_ind]['values'])],axis=0)
        else:
          print("new",f['token'])
          for layer_ind in np.arange(12):
            sent_embeddings_for_each_layer[layer_ind].append(np.asarray(f['layers'][layer_ind]['values']))

      real_ind += 1
  for layer_ind in np.arange(12):
    embeddings_for_each_layer[layer_ind].appned(np.asarray(sent_embeddings_for_each_layer[layer_ind]))