"""Main file to run for training and evaluating the models.

"""
import sys
sys.path.append('~/Codes/GoogleLM1b/')

from ExplainBrain import ExplainBrain
from read_dataset.harrypotter_data import HarryPotterReader
from computational_model.text_encoder import TfHubElmoEncoder, TfTokenEncoder, TfHubLargeUniversalEncoder, GloVeEncoder, GoogleLMEncoder
from mapping_models.sk_mapper import SkMapper

import tensorflow as tf
import numpy as np
import os


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('subject_id', 1, 'subject id')
tf.flags.DEFINE_integer('fold_id', 0, 'fold id')
tf.flags.DEFINE_list('delays',[-6,-4,-2,0] , 'delay list')
tf.flags.DEFINE_boolean('cross_delay', False, 'try different train and test delays')

tf.flags.DEFINE_float('alpha', 1, 'alpha')
tf.flags.DEFINE_string('embedding_dir', '/Users/iSam/Codes/Data/word_embeddings/glove.6B/glove.6B.300d.txt', 'path to the file containing the embeddings')
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

  print("brain data dir: ", hparams.brain_data_dir)
  harrypotter_clean_sentences = np.load(os.path.join(hparams.brain_data_dir,"harrypotter_cleaned_sentences.npy"))

  TextEncoderDic = {'elmo':TfHubElmoEncoder(hparams),
                    'tf_token': TfTokenEncoder(hparams),
                    'universal_large': TfHubLargeUniversalEncoder(hparams),
                    'glove': GloVeEncoder(hparams, harrypotter_clean_sentences),
                    'google_lm': GoogleLMEncoder(hparams, path= os.path.join(hparams.root,"GoogleLM1b/"))
                    }

  tf.logging.set_verbosity(tf.logging.INFO)

  # Define how we want to read the brain data
  print("1. initialize brain data reader ...")
  brain_data_reader = HarryPotterReader(data_dir=hparams.brain_data_dir)

  # Define how we want to computationaly represent the stimuli
  print("2. initialize text encoder ...")
  stimuli_encoder = TextEncoderDic[hparams.text_encoder]

  print("3. initialize mapper ...")
  mapper = (SkMapper, {'hparams':hparams})

  # Build the pip
  # eline object
  print("4. initialize Explainer...")
  explain_brain = ExplainBrain(hparams, brain_data_reader, stimuli_encoder, mapper)

  # Train and evaluate how well we can predict the brain activatiobs
  print("5. train and evaluate...")
  explain_brain.train_mappers(delays=hparams.delays, cross_delay=hparams.cross_delay,eval=True, fold_index=hparams.fold_id)