"""Main file to run for training and evaluating the models.

"""
import sys
sys.path.append('~/Codes/GoogleLM1b/')
import tensorflow as tf
import numpy as np
import os
import pickle
from ExplainBrain import ExplainBrain
from read_dataset.harrypotter_data import HarryPotterReader
from util.misc import get_dists, compute_dist_of_dists
from util.plotting import plot
from voxel_preprocessing.select_voxels import VarianceFeatureSelection
import tensorflow as tf
import numpy as np
import os

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('subject_id', 1, 'subject id')
tf.flags.DEFINE_integer('fold_id', 0, 'fold id')
tf.flags.DEFINE_list('delays',[-6,-4,-2,0] , 'delay list')
tf.flags.DEFINE_boolean('cross_delay', False, 'try different train and test delays')

tf.flags.DEFINE_float('alpha', 1, 'alpha')
tf.flags.DEFINE_string('embedding_dir', 'Data/word_embeddings/glove.6B/glove.6B.300d.txt', 'path to the file containing the embeddings')
tf.flags.DEFINE_string('brain_data_dir', 'Data/harrypotter/', 'Brain Data Dir')
tf.flags.DEFINE_string('root', '/Users/samiraabnar/Codes/', 'general path root')

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


tf.flags.DEFINE_string("emb_save_dir",'bridge_models/embeddings/', 'where to save embeddings')


if __name__ == '__main__':
  hparams = FLAGS
  print("roots", hparams.root)

  hparams.brain_data_dir = os.path.join(hparams.root, hparams.brain_data_dir)
  hparams.emb_save_dir = os.path.join(hparams.root, hparams.emb_save_dir)

  saving_dir = os.path.join(hparams.root, hparams.emb_save_dir,hparams.text_encoder +"_"+ hparams.embedding_type +"_"+ hparams.context_mode
                            + "_" + str(hparams.past_window) +"-"+ str(hparams.future_window) + "_onlypast-" + str(hparams.only_past))
  print("brain data dir: ", hparams.brain_data_dir)
  print("saving dir: ", saving_dir)

  brain_data_reader = HarryPotterReader(data_dir=hparams.brain_data_dir)

  story_features = np.load('story_features.npy').item()
  delay = 0
  blocks = [1,2,3,4]
  selected_steps = {}
  selecting_feature = 'end'


  embeddings = {}
  labels = {}

  encoder_types = ['google_lm','elmo','universal_large', 'glove'] #, 'google_lm','tf_token']
  embedding_types = {
    'universal_large': ['none'],
    'google_lm': ['lstm_0', 'lstm_1'],
    'glove': ['none'],
    'tf_token': ['none'],
    'elmo': ['lstm_outputs1','lstm_outputs2']
  }

  for encoder_type in encoder_types:
    for embedding_type in embedding_types[encoder_type]:
      embedding_key = encoder_type +"_"+embedding_type
      embeddings[embedding_key] = []
      labels[embedding_key] = []


      # Adding None context (only stimuli is given)


      # Adding sentence based contexts
      for context_size in np.arange(7):
        embeddings[embedding_key].append([])
        a = pickle.load(
          open(os.path.join(hparams.emb_save_dir,
                            encoder_type+'_'+embedding_type+'_sentence_' + str(context_size) + '-0_onlypast-True'),
               'rb'))
        time_steps = a['time_steps']
        selected_steps = {1:{},2:{},3:{},4:{}}
        for b in blocks:
          selected_steps[b][embedding_key+"_"+str(context_size)] = []
          for i in np.arange(len(time_steps[b])):
            current_step = time_steps[b][i]
            if selecting_feature == 'none':
              if i >= a['start_steps'][b] and i < a['end_steps'][b]:
                selected_steps[b][embedding_key +"_"+ str(context_size)].append(i)
            elif story_features[selecting_feature][b][current_step] == 1 and i >= a['start_steps'][b] and i < a['end_steps'][b]:
              selected_steps[b][embedding_key +"_"+ str(context_size)].append(i)

          print(a['start_steps'][b],a['end_steps'][b])
          embeddings[embedding_key][-1].extend(list(np.asarray(a['encoded_stimuli'][b])[selected_steps[b][embedding_key+"_"+str(context_size)]]))

        embeddings[embedding_key][-1] = np.asarray(embeddings[embedding_key][-1])
        print(embedding_key)
        print(embeddings[embedding_key][-1].shape)
        labels[embedding_key].append(embedding_key+'_context_' + str(context_size))


  print("####################################")
  all_embeddings_per_context_length = {}
  all_labels_per_context_length = {}
  for context_size in np.arange(7):
    all_embeddings_per_context_length[context_size] = []
    all_labels_per_context_length[context_size] = []
    for key in embeddings.keys():
      all_embeddings_per_context_length[context_size] += [embeddings[key][context_size]]
      all_labels_per_context_length[context_size] += [labels[key][context_size]]

  import csv

  for context_size in np.arange(7):
    x, C = get_dists(all_embeddings_per_context_length[context_size])
    klz, prz, labels_ = compute_dist_of_dists(x, C, all_labels_per_context_length[context_size])
    print(context_size,':',np.mean(prz))
    with open('com_sim_prz_' + str(context_size) + '.csv', 'w') as f:
     writer = csv.writer(f)
     writer.writerow(all_labels_per_context_length[context_size])
     writer.writerows(prz)

    plot(prz, all_labels_per_context_length[context_size])

