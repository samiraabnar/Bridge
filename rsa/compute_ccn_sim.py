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
from util.cca import get_cca_similarity
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
  delay = 2
  blocks = [1,2,3,4]


  voxel_to_regions = {}
  regions_to_voxels = {}
  brain_fs = {}
  for subject in [1,2,3,4,5,6,7,8]:
    voxel_to_regions[subject], regions_to_voxels[subject] = brain_data_reader.get_voxel_to_region_mapping(subject_id=subject)
    brain_fs[subject] = VarianceFeatureSelection()

  best_brain_regions = ['Postcentral_L',
                        'Temporal_Inf_R',
                        'Cerebelum_Crus1_L',
                        'Fusiform_R',
                        'Temporal_Pole_Mid_L',
                        'Pallidum_L',
                        'Temporal_Mid_R',
                        'Temporal_Pole_Sup_L',
                        'Putamen_R',
                        'ParaHippocampal_L',
                        'Hippocampus_R',
                        'Amygdala_L',
                        'Postcentral_R',
                        'Thalamus_R',
                        'Precentral_R',
                        'Parietal_Sup_L' ]



  embeddings = {}
  labels = {}
  #Load Elmo:

  #Load Glove Embeddings

  #Load Google LM

  #Load Universal Embeddings
  encoder_types = ['google_lm'] #, 'elmo','universal_large', 'glove', 'google_lm','tf_token']
  embedding_types = {
    'universal_large': ['none'],
    'google_lm': ['lstm_0', 'lstm_1'],
    'glove': ['none'],
    'tf_token': ['none'],
    'elmo': ['word_emb','lstm_outputs1','lstm_outputs2', 'elmo']
  }

  for encoder_type in encoder_types:
    for embedding_type in embedding_types[encoder_type]:
      embedding_key = encoder_type +"_"+embedding_type
      embeddings[embedding_key] = []
      labels[embedding_key] = []
      for context_size in np.arange(7):
        embeddings[embedding_key].append([])
        a = pickle.load(
          open(os.path.join(hparams.emb_save_dir,
                            encoder_type+'_'+embedding_type+'_sentence_' + str(context_size) + '-0_onlypast-True'),
               'rb'))

        for b in blocks:
          print(a['start_steps'][b],a['end_steps'][b])
          embeddings[embedding_key][-1].extend(list(a['encoded_stimuli'][b][a['start_steps'][b]:a['end_steps'][b]]))

        embeddings[embedding_key][-1] = np.asarray(embeddings[embedding_key][-1])
        print(embedding_key)
        print(embeddings[embedding_key][-1].shape)
        labels[embedding_key].append(embedding_key+'_context_' + str(context_size))

  all_embeddings = []
  all_labels = []
  for key in embeddings.keys():
    all_embeddings += embeddings[key]
    all_labels += labels[key]


  brain_regions = []
  brain_regions_labels = []
  #Load Brains
  brains = []
  brain_labels = []
  for i in np.arange(1, 9):
    brains.append([])
    a = pickle.load(open(os.path.join(hparams.emb_save_dir,str(i) + '_Brain'),
                         'rb'))
    for b in blocks:
      brains[-1].extend(a['brain_activations'][b][11+delay:-1])


    brains[-1] = np.asarray(brains[-1])
    brain_fs[i].fit(brains[-1])
    original_selected_voxels = brain_fs[i].get_selected_indexes()
    print(len(original_selected_voxels))

    for region_name, voxels in regions_to_voxels[i].items():
      if region_name in best_brain_regions:
        voxels = [v for v in voxels if v in original_selected_voxels]
        brain_regions.append(brains[-1][:, voxels])
        brain_regions_labels.append(['subject_'+str(i)+region_name])

    selected_voxels = [v for v in original_selected_voxels if voxel_to_regions[i][v] in best_brain_regions]
    brains[-1] = brains[-1][:, selected_voxels]
    brain_labels.append('brain_' + str(i))
    print(brains[-1].shape)

  print(all_embeddings[0].shape)
  sim_dic_coef1 = {}
  sim_dic_coef2 = {}
  for key in embeddings.keys():
    for context_size in [1]:
      new_key = key+str(context_size)
      sim_dic_coef1[new_key] = []
      sim_dic_coef2[new_key] = []
      for sub in np.arange(9):
        print(brains[sub].shape)
        print(embeddings[key][context_size].shape)
        c = get_cca_similarity(np.transpose(embeddings[key][context_size]), np.transpose(brains[sub]))
        print(new_key," ",sub, np.mean(c['mean']))
        sim_dic_coef1[new_key].append(np.mean(c['mean'][0]))
        sim_dic_coef2[new_key].append(np.mean(c['mean'][1]))

  for k in sim_dic_coef1:
    print(k," ",sim_dic_coef1[k], sim_dic_coef2[k])

  print("*********************")
  print(c.keys())
  print(len(c['cca_coef1']))
  print(len(c['cca_coef2']))
  print(c['mean'])

