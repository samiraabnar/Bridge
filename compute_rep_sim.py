"""Main file to run for training and evaluating the models.

"""
import sys
sys.path.append('~/Codes/GoogleLM1b/')
import tensorflow as tf
import numpy as np
import os
import pickle

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('subject_id', 1, 'subject id')
tf.flags.DEFINE_integer('fold_id', 0, 'fold id')
tf.flags.DEFINE_list('delays',[-6,-4,-2,0] , 'delay list')
tf.flags.DEFINE_boolean('cross_delay', False, 'try different train and test delays')

tf.flags.DEFINE_float('alpha', 1, 'alpha')
tf.flags.DEFINE_string('embedding_dir', 'Data/word_embeddings/glove.6B/glove.6B.300d.txt', 'path to the file containing the embeddings')
tf.flags.DEFINE_string('brain_data_dir', '/Users/iSam/Codes/Data/harrypotter/', 'Brain Data Dir')
tf.flags.DEFINE_string('root', '/Users/iSam/Codes/', 'general path root')

tf.flags.DEFINE_enum('text_encoder', 'glove',
                     ['glove','elmo', 'tf_token' ,'universal_large', 'google_lm'], 'which encoder to use')
tf.flags.DEFINE_string('embedding_type', 'lstm_outputs1', 'ELMO: word_emb, lstm_outputs1, lstm_outputs2 ')
tf.flags.DEFINE_string('context_mode', 'none', 'type of context (sentence, block, none)')
tf.flags.DEFINE_integer('past_window', 3, 'window size to the past')
tf.flags.DEFINE_integer('future_window', 0, 'window size to the future')
tf.flags.DEFINE_boolean('only_past', True, 'window size to the future')


tf.flags.DEFINE_boolean('save_data', True ,'save data flag')
tf.flags.DEFINE_boolean('load_data', True ,'load data flag')
tf.flags.DEFINE_boolean('save_encoded_stimuli', True, 'save encoded stimuli')
tf.flags.DEFINE_boolean('load_encoded_stimuli', True, 'load encoded stimuli')

tf.flags.DEFINE_boolean('save_models', True ,'save models flag')

tf.flags.DEFINE_string("param_set", None, "which param set to use")

def basic_glove_params(hparams):
  hparams.context_mode = 'none'
  hparams.text_encoder = 'glove'
  hparams.alpha = 1.0

  return hparams

def sentence_glove_params(hparams):
  hparams.context_mode = 'sentence'
  hparams.text_encoder = 'glove'
  hparams.alpha = 1.0
  hparams.delays = [-6,-4,-2,0]

  return hparams

def basic_elmo_params(hparams):
  hparams.context_mode = 'none'
  hparams.text_encoder = 'elmo'
  hparams.alpha = 1.0

  return hparams

def sentence_elmo_params(hparams):
  hparams.context_mode = 'sentence'
  hparams.text_encoder = 'elmo'
  hparams.alpha = 1.0

  return hparams



hparams = FLAGS
if __name__ == '__main__':
  if hparams.param_set is 'glove':
    hparams = basic_glove_params(hparams)
  elif hparams.param_set is 'basic_elmo':
    hparams = basic_elmo_params(hparams)
  elif hparams.param_set is 'sentence_elmo':
    hparams = sentence_elmo_params(hparams)
  print("***********")
  print(hparams)
  print("***********")

  hparams.embedding_dir = os.path.join(hparams.root, hparams.embedding_dir)
  harrypotter_clean_sentences = np.load(os.path.join(hparams.brain_data_dir,"harrypotter_cleaned_sentences.npy"))


  #Load Elmo:

  #Load Glove Embeddings

  #Load Google LM

  #Load Universal Embeddings
  universal_embeddings = []
  uni_labels = []
  for context_size in np.arange(7):
    universal_embeddings.append([])
    a = pickle.load(
      open('gdrive/My Drive/harrypotter/universal_large_none_sentence_' + str(context_size) + '-0_onlypast-True', 'rb'))
    universal_embeddings[-1].extend(a['encoded_stimuli'][1])
    universal_embeddings[-1].extend(a['encoded_stimuli'][2])
    universal_embeddings[-1].extend(a['encoded_stimuli'][3])
    universal_embeddings[-1].extend(a['encoded_stimuli'][4])

    universal_embeddings[-1] = np.asarray(universal_embeddings[-1])
    uni_labels.append('uni_context_' + str(context_size))
    print(universal_embeddings[-1].shape)


  #Load Brains
  brains = []
  brain_labels = []
  for i in np.arange(1, 9):
    brains.append([])
    a = pickle.load(open('/content/gdrive/My Drive/harrypotter/' + str(i) + '_Brain', 'rb'))
    brains[-1].extend(a['brain_activations'][1])
    brains[-1].extend(a['brain_activations'][2])
    brains[-1].extend(a['brain_activations'][3])
    brains[-1].extend(a['brain_activations'][4])

    brains[-1] = np.asarray(brains[-1])
    brain_labels.append('brain_' + str(i))
    print(brains[-1].shape)


  #Get brain regions:
