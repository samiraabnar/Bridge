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
tf.flags.DEFINE_string('root', '/Users/samiraabnar/Codes/', 'general path root')

tf.flags.DEFINE_enum('text_encoder', 'google_lm',
                     ['glove','elmo', 'tf_token' ,'universal_large', 'google_lm'], 'which encoder to use')
tf.flags.DEFINE_string('embedding_type', 'lstm_1', 'ELMO: word_emb, lstm_outputs1, lstm_outputs2, elmo ')
tf.flags.DEFINE_string('context_mode', 'sentence', 'type of context (sentence, block, none)')
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

  for i in [0,1,2,3,4,5,6]:
    b = np.load('../bridge_data/processed/harrypotter/1_sentence_window'+str(i)+'-0_stimuli_in_context.npy').item()
    a = pickle.load(open('../bridge_models/embeddings/google_lm_'+hparams.embedding_type+'_sentence_'+str(i)+'-0_onlypast-True', 'rb'))
    print(a['encoded_stimuli'][1].shape)
    if len(a['encoded_stimuli'][1].shape) < 2:
      for block in [1,2,3,4]:
        c = []
        for item in np.arange(len(b[block])):
          if b[block][item][1] is None:
            c.append(np.zeros(1024))
          else:
            c.append(np.mean(np.asarray(a['encoded_stimuli'][block][item])[b[block][item][1]], axis=0)[0])
        a['encoded_stimuli'][block] = c


    saving_dic = a #{'time_steps':time_steps, 'start_steps':start_steps, 'end_steps':end_steps, 'encoded_stimuli':encoded_stimuli, 'stimuli':stimuli}
    pickle.dump(saving_dic, open('../bridge_models/embeddings/google_lm_'+hparams.embedding_type+'_sentence_'+str(i)+'-0_onlypast-True', 'wb'))
    saving_dic = pickle.load(open('../bridge_models/embeddings/google_lm_'+hparams.embedding_type+'_sentence_'+str(i)+'-0_onlypast-True', 'rb'))

