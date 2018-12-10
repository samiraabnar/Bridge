"""
save encoded stimuli ...
"""

import sys

sys.path.append('~/Codes/GoogleLM1b/')
sys.path.append('~/Codes/bert/')

from ExplainBrain import ExplainBrain
from read_dataset.harrypotter_data import HarryPotterReader
from computational_model.text_encoder import BertEncoder

import tensorflow as tf
import numpy as np
import os
import pickle
from tqdm import tqdm

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('subject_id', 1, 'subject id')
tf.flags.DEFINE_integer('fold_id', 0, 'fold id')
tf.flags.DEFINE_list('delays',[-6,-4,-2,0] , 'delay list')
tf.flags.DEFINE_boolean('cross_delay', False, 'try different train and test delays')

tf.flags.DEFINE_float('alpha', 1, 'alpha')
tf.flags.DEFINE_string('embedding_dir', 'Data/word_embeddings/glove.6B/glove.6B.300d.txt', 'path to the file containing the embeddings')
tf.flags.DEFINE_string('brain_data_dir', 'Data/harrypotter/', 'Brain Data Dir')
tf.flags.DEFINE_string('root', '/Users/samiraabnar/Codes/', 'general path root')

tf.flags.DEFINE_enum('text_encoder', 'bert',
                     ['glove','elmo', 'tf_token' ,'universal_large', 'google_lm', 'bert'], 'which encoder to use')
tf.flags.DEFINE_string('embedding_type', '', 'ELMO: word_emb, lstm_outputs1, lstm_outputs2, elmo ')
tf.flags.DEFINE_string('context_mode', 'sentence', 'type of context (sentence, block, none)')
tf.flags.DEFINE_integer('past_window', 6, 'window size to the past')
tf.flags.DEFINE_integer('future_window', 0, 'window size to the future')
tf.flags.DEFINE_boolean('only_past', True, 'window size to the future')


tf.flags.DEFINE_boolean('save_data', True ,'save data flag')
tf.flags.DEFINE_boolean('load_data', True ,'load data flag')
tf.flags.DEFINE_boolean('save_encoded_stimuli', True, 'save encoded stimuli')
tf.flags.DEFINE_boolean('load_encoded_stimuli', False, 'load encoded stimuli')

tf.flags.DEFINE_boolean('save_models', True ,'save models flag')

tf.flags.DEFINE_string("param_set", None, "which param set to use")
tf.flags.DEFINE_string("emb_save_dir",'bridge_models/embeddings/', 'where to save embeddings')
tf.flags.DEFINE_string('layers', '0,1,2,3,4,5,6,7,8,9,10,11', 'layers')

if __name__ == '__main__':
  hparams = FLAGS
  print("roots", hparams.root)

  hparams.embedding_dir = os.path.join(hparams.root, hparams.embedding_dir)
  hparams.brain_data_dir = os.path.join(hparams.root, hparams.brain_data_dir)

  saving_dir = os.path.join(hparams.root, hparams.emb_save_dir,hparams.text_encoder +"_"+ hparams.embedding_type +"_"+ hparams.context_mode
                            + "_" + str(hparams.past_window) +"-"+ str(hparams.future_window) + "_onlypast-" + str(hparams.only_past))
  print("brain data dir: ", hparams.brain_data_dir)
  print("saving dir: ", saving_dir)


  TextEncoderDic = {'bert':BertEncoder(hparams),
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




  # Encode the stimuli and get the representations from the computational model.
  tf.logging.info('Encoding the stimuli ...')


  def integration_fn(inputs, axis, max_size=200000):
    if len(inputs.shape) > 1:
      inputs = np.mean(inputs, axis=axis)
    size = inputs.shape[-1]
    return inputs[:np.min([max_size, size])]


  bert_encoder_obj = BertEncoder(hparams)

  encoded_stimuli_per_each_layer = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {},
                               7: {}, 8: {}, 9: {}, 10: {}, 11: {}}

  embeddings_for_each_layer = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {},
                               7: {}, 8: {}, 9: {}, 10: {}, 11: {}}

  print("#########")
  sentences = {1:[],2:[],3:[],4:[]}
  target_indexes = {1:[],2:[],3:[],4:[]}
  sentences_lengths = {1:[],2:[],3:[],4:[]}
  output_embeddings = {}
  for block in [1, 2, 3, 4]:
    for stim, index in stimuli[block]:
      sentences[block].append(' '.join(stim))
      target_indexes[block].append(index)
      sentences_lengths[block].append(len(stim))
    for layer_ind in np.arange(12):
      encoded_stimuli_per_each_layer[layer_ind][block] = []
      embeddings_for_each_layer[layer_ind][block] = []

    output_embeddings[block] = bert_encoder_obj.get_embeddings_values(sentences[block], sentences_lengths[block])

    for sent_len, sent,  index, output in tqdm(zip(sentences_lengths[block], sentences[block],target_indexes[block], output_embeddings[block])):
      print("len sent",len(sent), len(output))
      sent_embeddings_for_each_layer = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [],
                                        7: [], 8: [], 9: [], 10: [], 11: []}
      current_outputs = []
      real_ind = 0
      while real_ind < len(output):
        f = output[real_ind]
        if f['token'] != '[CLS]' and f['token'] != '[SEP]':
          if f['token'].startswith("##"):
            for layer_ind in np.arange(12):
              sent_embeddings_for_each_layer[layer_ind][-1] = np.mean(
                [sent_embeddings_for_each_layer[layer_ind][-1], np.asarray(f['layers'][layer_ind]['values'])], axis=0)
          else:
            for layer_ind in np.arange(12):
              sent_embeddings_for_each_layer[layer_ind].append(np.asarray(f['layers'][layer_ind]['values']))

        real_ind += 1
      for layer_ind in np.arange(12):
        print(np.asarray(sent_embeddings_for_each_layer[layer_ind]).shape)
        embeddings_for_each_layer[layer_ind][block].append(np.asarray(sent_embeddings_for_each_layer[layer_ind]))
        encoded_stimuli_per_each_layer[layer_ind][block].append(np.asarray(sent_embeddings_for_each_layer[layer_ind])[index])

  for layer_ind in np.arange(12):
    print(embeddings_for_each_layer[layer_ind][1][11].shape)
    print(encoded_stimuli_per_each_layer[layer_ind][1][11].shape)

    saving_dic = {'time_steps':time_steps, 'start_steps':start_steps, 'end_steps':end_steps, 'encoded_stimuli':encoded_stimuli_per_each_layer[layer_ind], 'stimuli':stimuli}
    pickle.dump(saving_dic, open(saving_dir+'_'+str(layer_ind), 'wb'))
    
    saving_dic = {'time_steps':time_steps, 'start_steps':start_steps, 'end_steps':end_steps, 'stimuli_embeddings': embeddings_for_each_layer[layer_ind], 'stimuli':stimuli}
    pickle.dump(saving_dic, open(saving_dir+'_all_'+str(layer_ind), 'wb'))
