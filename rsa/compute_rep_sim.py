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
tf.flags.DEFINE_list('delay', 0 , 'delay list')
tf.flags.DEFINE_list('blocks', [4], 'experiment/story blocks to consider')


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

def read_embeddings(a, blocks, selecting_feature, context_mode, selected_steps, delay):
  time_steps = a['time_steps']
  print("blocks:", blocks)
  embedded_steps = []
  for b in blocks:
    print("read embedding block:", b)
    selected_steps[b][embedding_key + context_mode] = []
    for i in np.arange(len(time_steps[b])):
      current_step = time_steps[b][i]
      if selecting_feature == 'none':
        if i >= a['start_steps'][b] and i < a['end_steps'][b] and i+delay > 0 and i+delay < len(a['encoded_stimuli'][b]):
          selected_steps[b][embedding_key + context_mode].append(i)
      elif story_features[selecting_feature][b][current_step] == 1 and i >= a['start_steps'][b] and i < \
          a['end_steps'][b] and i+delay > 0 and i+delay < len(a['encoded_stimuli'][b]):
        selected_steps[b][embedding_key + context_mode].append(i)

      if len(np.asarray(a['encoded_stimuli'][b][i]).shape)>1:
        a['encoded_stimuli'][b][i] = np.mean(a['encoded_stimuli'][b][i], axis=0)

    embedded_steps.extend(list(np.asarray(a['encoded_stimuli'][b])[selected_steps[b][embedding_key + context_mode]]))
    print(a['start_steps'][b], a['end_steps'][b])

  return embedded_steps, selected_steps



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

  story_features = np.load('../story_features.npy').item()
  delay = -1
  selected_steps = {}
  print(story_features.keys())
  selecting_feature = 'none' # none | end | ch | quote

  voxel_to_regions = {}
  regions_to_voxels = {}
  brain_fs = {}
  for subject in np.arange(1,9):
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

  encoder_types = ['google_lm','elmo','universal_large', 'glove','bert'] #, 'google_lm','tf_token']
  embedding_types = {
    'universal_large': ['none'],
    'google_lm': ['lstm_0', 'lstm_1'],
    'glove': ['none'],
    'elmo': ['lstm_outputs1','lstm_outputs2'],
    'bert': ['0', '6','11'],
    'tf_token': ['none']
  }

  selected_steps = {1: {}, 2: {}, 3: {}, 4: {}}
  # Read the representations from the computational models
  for encoder_type in encoder_types:
    for embedding_type in embedding_types[encoder_type]:
      embedding_key = encoder_type +"_"+embedding_type
      embeddings[embedding_key] = []
      labels[embedding_key] = []
      embeddings[embedding_key].append([])

      # read embedding with no additional context!
      if encoder_type == 'bert':
        a = pickle.load(
          open(os.path.join(hparams.emb_save_dir,
                            encoder_type + '_' + '_none_' + str(0) + '-0_onlypast-True_'+embedding_type),
               'rb'))
      else:
        a = pickle.load(
          open(os.path.join(hparams.emb_save_dir,
                            encoder_type + '_' + embedding_type + '_none_' + str(0) + '-0_onlypast-True'),
               'rb'))

      embedded_steps, selected_steps = read_embeddings(a,FLAGS.blocks, selecting_feature,
                                                       context_mode="_none", selected_steps=selected_steps, delay=delay)

      embeddings[embedding_key][-1].extend(embedded_steps)
      embeddings[embedding_key][-1] = np.asarray(embeddings[embedding_key][-1])
      labels[embedding_key].append(embedding_key + '_context_none')


      # read embedding with context size from 0-to7 sentences;
      # zero means current sentence up to after the target words.
      for context_size in np.arange(7):
        embeddings[embedding_key].append([])
        if encoder_type == 'bert':
          a = pickle.load(
            open(os.path.join(hparams.emb_save_dir,
                              encoder_type + '_none'+ '_sentence_' + str(context_size) + '-0_onlypast-True_' + embedding_type),
                 'rb'))
        else:
          a = pickle.load(
            open(os.path.join(hparams.emb_save_dir,
                              encoder_type+'_'+embedding_type+'_sentence_' + str(context_size) + '-0_onlypast-True'),
                 'rb'))

        embedded_steps, selected_steps = read_embeddings(a, FLAGS.blocks, selecting_feature,
                                                         context_mode="_"+str(context_size),
                                                         selected_steps=selected_steps, delay=delay)

        print("shape:" ,np.asarray(embedded_steps).shape)
        embeddings[embedding_key][-1].extend(embedded_steps)
        embeddings[embedding_key][-1] = np.asarray(embeddings[embedding_key][-1])
        labels[embedding_key].append(embedding_key+'_context_' + str(context_size))

  brain_selected_steps = {1: [], 2: [], 3: [], 4: []}
  for b in [1, 2, 3, 4]:
    for sel in selected_steps[b]:
      brain_selected_steps[b] = list(np.asarray(selected_steps[b][sel]) + delay)

  #Load Brains
  brains = []
  brain_labels = []
  brain_regions = []
  brain_regions_labels = []
  for i in np.arange(1, 9):
    brains.append([])
    a = pickle.load(open(os.path.join(hparams.emb_save_dir, str(i) + '_Brain'),
                         'rb'))
    for b in FLAGS.blocks:
      brains[-1].extend(a['brain_activations'][b][brain_selected_steps[b]])

    brains[-1] = np.asarray(brains[-1])
    brain_fs[i].fit(brains[-1])
    original_selected_voxels = brain_fs[i].get_selected_indexes()

    for region_name, voxels in regions_to_voxels[i].items():
      if region_name in best_brain_regions:
        voxels = [v for v in voxels if v in original_selected_voxels]
        brain_regions.append(brains[-1][:, voxels])
        brain_regions_labels.append('subject_'+str(i)+region_name)


    selected_voxels = [v for v in original_selected_voxels if voxel_to_regions[i][v] in best_brain_regions]
    brains[-1] = brains[-1][:, selected_voxels]
    brain_labels.append('brain_' + str(i))

  all_embeddings = []
  all_labels = []
  for key in embeddings.keys():
    all_embeddings += embeddings[key]
    all_labels += labels[key]


  uniform_random_300 = np.random.uniform(low=-1, high=1, size=(all_embeddings[0].shape[0],300))
  normal_random_300 = np.random.standard_normal(size=(all_embeddings[0].shape[0],300))
  uniform_random_100 = np.random.uniform(low=-1, high=1, size=(all_embeddings[0].shape[0], 100))
  normal_random_100 = np.random.standard_normal(size=(all_embeddings[0].shape[0], 100))
  uniform_random_1000 = np.random.uniform(low=-1, high=1, size=(all_embeddings[0].shape[0], 1000))
  normal_random_1000 = np.random.standard_normal(size=(all_embeddings[0].shape[0], 1000))


  randoms = [uniform_random_300, normal_random_300,
             uniform_random_100, normal_random_100,
             uniform_random_1000, normal_random_1000]

  random_labels = ['uniform_random_300', 'normal_ransom_300',
                   'uniform_random_100', 'normal_random_100',
                   'uniform_random_1000', 'normal_random_1000']

  print("brain shapes:", brain_regions[0].shape)
  x, C = get_dists(brain_regions + all_embeddings + randoms)
  klz, prz, labels_, p_vals = compute_dist_of_dists(x, C, brain_regions_labels + all_labels + random_labels)
  plot(prz, labels_)
  plot(p_vals, labels_)

  klz = np.asarray(klz)
  print(klz.shape)

  dic2save = {"klz": klz,
  "prz": prz,
  "labels_": labels_,
  "p_vals": p_vals}
  np.save('_'.join(list(map(str,FLAGS.blocks)))+"rsa_results_all_brain_regions_"+str(delay)+"_"+selecting_feature, dic2save)

  x, C = get_dists(brains + all_embeddings + randoms)
  klz, prz, labels_, p_vals = compute_dist_of_dists(x, C, brain_labels + all_labels + random_labels)
  plot(prz, labels_)
  plot(p_vals, labels_)

  klz = np.asarray(klz)
  print(klz.shape)

  dic2save = {"klz": klz,
              "prz": prz,
              "labels_": labels_,
              "p_vals": p_vals}
  np.save('_'.join(list(map(str,FLAGS.blocks)))+"rsa_results_all_full_brain_"+str(delay)+"_"+selecting_feature, dic2save)
