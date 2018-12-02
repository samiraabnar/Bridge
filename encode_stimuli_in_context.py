"""
save encoded stimuli ...
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
import pickle

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('subject_id', 1, 'subject id')
tf.flags.DEFINE_integer('fold_id', 0, 'fold id')
tf.flags.DEFINE_list('delays',[-6,-4,-2,0] , 'delay list')
tf.flags.DEFINE_boolean('cross_delay', False, 'try different train and test delays')

tf.flags.DEFINE_float('alpha', 1, 'alpha')
tf.flags.DEFINE_string('embedding_dir', 'Data/word_embeddings/glove.6B/glove.6B.300d.txt', 'path to the file containing the embeddings')
tf.flags.DEFINE_string('brain_data_dir', 'Data/harrypotter/', 'Brain Data Dir')
tf.flags.DEFINE_string('root', '/Users/iSam/Codes/', 'general path root')

tf.flags.DEFINE_enum('text_encoder', 'universal_large',
                     ['glove','elmo', 'tf_token' ,'universal_large', 'google_lm'], 'which encoder to use')
tf.flags.DEFINE_string('embedding_type', '', 'ELMO: word_emb, lstm_outputs1, lstm_outputs2, elmo ')
tf.flags.DEFINE_string('context_mode', 'none', 'type of context (sentence, block, none)')
tf.flags.DEFINE_integer('past_window', 3, 'window size to the past')
tf.flags.DEFINE_integer('future_window', 0, 'window size to the future')
tf.flags.DEFINE_boolean('only_past', True, 'window size to the future')


tf.flags.DEFINE_boolean('save_data', True ,'save data flag')
tf.flags.DEFINE_boolean('load_data', True ,'load data flag')
tf.flags.DEFINE_boolean('save_encoded_stimuli', True, 'save encoded stimuli')
tf.flags.DEFINE_boolean('load_encoded_stimuli', False, 'load encoded stimuli')

tf.flags.DEFINE_boolean('save_models', True ,'save models flag')

tf.flags.DEFINE_string("param_set", None, "which param set to use")
tf.flags.DEFINE_string("emb_save_dir",'bridge_models/embeddings/', 'where to save embeddings')

if __name__ == '__main__':
  hparams = FLAGS
  print("roots", hparams.root)

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

  tf.logging.set_verbosity(tf.logging.INFO)

  # Define how we want to read the brain data
  print("1. initialize brain data reader ...")
  brain_data_reader = HarryPotterReader(data_dir=hparams.brain_data_dir)

  # Define how we want to computationaly represent the stimuli
  print("2. initialize text encoder ...")
  stimuli_encoder = TextEncoderDic[hparams.text_encoder]

  # Explain Brain object with no mapper
  print("4. initialize Explainer...")
  explain_brain = ExplainBrain(hparams, brain_data_reader, stimuli_encoder, None)



  # Load the brain data
  tf.logging.info('Loading brain data ...')
  time_steps, brain_activations, stimuli, start_steps, end_steps = explain_brain.load_brain_experiment()

  tf.logging.info('Blocks: %s' % str(explain_brain.blocks))
  print('Example Stimuli %s' % str(stimuli[1][0]))

  # Preprocess brain activations
  brain_activations = explain_brain.preprocess_brain_activations(brain_activations,
                                                        voxel_preprocessings=explain_brain.voxel_preprocess(),
                                                        start_steps=start_steps, end_steps=end_steps,
                                                        resting_norm=False)

  # Encode the stimuli and get the representations from the computational model.
  tf.logging.info('Encoding the stimuli ...')


  def integration_fn(inputs, axis, max_size=200000):
    if len(inputs.shape) > 1:
      inputs = np.mean(inputs, axis=axis)
    size = inputs.shape[-1]
    return inputs[:np.min([max_size, size])]


  encoded_stimuli = explain_brain.encode_stimuli(stimuli, integration_fn=integration_fn)

  print("shape of encoded:",np.asarray(encoded_stimuli[1]).shape)
  saving_dic = {'time_steps':time_steps, 'start_steps':start_steps, 'end_steps':end_steps, 'encoded_stimuli':encoded_stimuli, 'stimuli':stimuli}
  pickle.dump(saving_dic, open(saving_dir, 'wb'))

