"""Main file to run for training and evaluating the models.

"""
import sys
sys.path.append('~/Codes/GoogleLM1b/')
import tensorflow as tf
import numpy as np
import os
import pickle
from ExplainBrain import ExplainBrain
from data_readers.harrypotter_data import HarryPotterReader
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


def compare_brains_in_regions(brain_regions, regions_to_voxels):
  for region_name, voxels in regions_to_voxels[i].items():
    if region_name in best_brain_regions:
      if region_name not in brain_regions:
        brain_regions[region_name] = []
      voxels = [v for v in voxels if v in brain_fs[i].get_selected_indexes()]
      brain_regions[region_name].append(brains[-1][:, voxels])

  for region_name in brain_regions:
    if min(map(len, brain_regions[region_name])) > 0:
      x_brain, C_brain = get_dists(brain_regions[region_name])
      klz_brain, labels_brain = compute_dist_of_dists(x_brain, C_brain, brain_labels)
      print(region_name, '\t', np.mean(klz_brain), '\t', np.std(klz_brain))
      plot(klz_brain, labels_brain)


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
  delay = 2 #2,0,3,1
  blocks = [1,2,3,4]
  selected_steps = {}
  selecting_feature = 'quote'



  embeddings = {}
  labels = {}

  encoder_types = ['bert']#, 'google_lm','tf_token']
  embedding_types = {
    'bert': ['none']
  }

  layers = [0,11]
  for encoder_type in encoder_types:
    for embedding_type in layers:
      embedding_type = str(embedding_type)
      embedding_key = encoder_type +"_"+embedding_type
      embeddings[embedding_key] = []
      labels[embedding_key] = []
      embeddings[embedding_key].append([])
      a = pickle.load(
        open(os.path.join(hparams.emb_save_dir,
                          encoder_type + '_' + '' + '_none_' + str(0) + '-0_onlypast-True_'+embedding_type),
             'rb'))
      time_steps = a['time_steps']
      selected_steps = {1: {}, 2: {}, 3: {}, 4: {}}
      for b in blocks:
        selected_steps[b][embedding_key + "_none" ] = []
        for i in np.arange(len(time_steps[b])):
          current_step = time_steps[b][i]
          if selecting_feature == 'none':
            if i >= a['start_steps'][b] and i < a['end_steps'][b]:
              selected_steps[b][embedding_key + "_none"].append(i)
          elif story_features[selecting_feature][b][current_step] == 1 and i >= a['start_steps'][b] and i < \
              a['end_steps'][b]:
            selected_steps[b][embedding_key + "_none"].append(i)
          a['encoded_stimuli'][b][i] = np.mean(a['encoded_stimuli'][b][i], axis=0)
        print(a['start_steps'][b], a['end_steps'][b])
        embeddings[embedding_key][-1].extend(
          list(np.asarray(a['encoded_stimuli'][b])[selected_steps[b][embedding_key + "_none"]]))

      embeddings[embedding_key][-1] = np.asarray(embeddings[embedding_key][-1])
      print(embedding_key)
      print(embeddings[embedding_key][-1].shape)
      labels[embedding_key].append(embedding_key + '_context_none')

      for context_size in np.arange(7):
        embeddings[embedding_key].append([])
        a = pickle.load(
          open(os.path.join(hparams.emb_save_dir,
                            encoder_type+'_'+'none'+'_sentence_' + str(context_size) + '-0_onlypast-True_'+embedding_type),
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
            a['encoded_stimuli'][b][i] = np.mean(a['encoded_stimuli'][b][i], axis=0)
          print(a['start_steps'][b],a['end_steps'][b])
          embeddings[embedding_key][-1].extend(list(np.asarray(a['encoded_stimuli'][b])[selected_steps[b][embedding_key+"_"+str(context_size)]]))

        embeddings[embedding_key][-1] = np.asarray(embeddings[embedding_key][-1])
        print(embedding_key)
        print(embeddings[embedding_key][-1].shape)
        labels[embedding_key].append(embedding_key+'_context_' + str(context_size))

  print("selected steps (they should be the same:")
  brain_selected_steps = {1:[],2:[],3:[],4:[]}
  for b in [1,2,3,4]:
    for sel in selected_steps[b]:
      print(sel, selected_steps[b][sel])
      brain_selected_steps[b] = list(np.asarray(selected_steps[b][sel])+delay)

  print("for brain: ", brain_selected_steps[1])
  print("####################################")
  all_embeddings = []
  all_labels = []
  for key in embeddings.keys():
    all_embeddings += embeddings[key][1:]
    all_labels += labels[key][1:]

  #x, C = get_dists(brains + all_embeddings)
  #klz, prz, labels_ = compute_dist_of_dists(x, C, brain_labels + all_labels)
  #plot(prz, brain_regions_labels)
  #klz = np.asarray(klz)
  #print(klz.shape)

  #print(brain_regions_labels)

  import csv

  x, C = get_dists(all_embeddings)
  klz, prz, labels_ = compute_dist_of_dists(x, C, all_labels)
  with open('bert_com_sim_prz_' + '.csv', 'w') as f:
   writer = csv.writer(f)
   writer.writerow(all_labels)
   writer.writerows(prz)

  print(all_labels)
  plot(prz, all_labels)

  half = int(len(all_labels)/2)
  for i in np.arange(half):
    print(all_labels[i].split("_")[-1], all_labels[i+half].split("_")[-1], prz[i][i+half])


