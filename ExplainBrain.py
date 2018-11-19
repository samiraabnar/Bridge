"""Pipeline for predicting brain activations.
"""

from evaluation.metrics import  mean_explain_variance
from util.misc import get_folds
from voxel_preprocessing.preprocess_voxels import detrend
from voxel_preprocessing.preprocess_voxels import minus_average_resting_states
from voxel_preprocessing.preprocess_voxels import reduce_mean
from voxel_preprocessing.select_voxels import VarianceFeatureSelection
from voxel_preprocessing.select_voxels import TopkFeatureSelection
from language_preprocessing.tokenize import SpacyTokenizer

import tensorflow as tf
import numpy as np
import itertools
from tqdm import tqdm
import os
import pickle
from pathlib import Path


class ExplainBrain(object):
  def __init__(self, hparams, brain_data_reader, stimuli_encoder, mapper_tuple):
    """

    :param brain_data_reader:
    :param stimuli_encoder:
    :param mapper_tuple: (mapper constructor, args)
    :param embedding_type:
    :param subject_id:
    """
    self.hparams = hparams
    self.brain_data_reader = brain_data_reader
    self.stimuli_encoder = stimuli_encoder
    self.mapper_tuple = mapper_tuple
    self.blocks = None
    self.folds = None
    self.subject_id = self.hparams.subject_id

    self.voxel_selectors = [VarianceFeatureSelection()]
    self.post_train_voxel_selectors = [TopkFeatureSelection()]
    self.embedding_type = self.hparams.embedding_type
    self.past_window = self.hparams.past_window
    self.future_window = self.hparams.future_window
    self.data_dir = "../bridge_data/processed/harrypotter/" + str(self.subject_id) + "_" + self.hparams.context_mode + "_window"+str(self.past_window)+"-"+str(self.future_window) + "_"
    self.model_dir = os.path.join("../bridge_models/", str(self.subject_id) + "_" + str(self.embedding_type)+"_"+self.hparams.context_mode+"_window"+str(self.past_window)+"-"+str(self.future_window))

  def load_brain_experiment(self, voxel_preprocessings=[], save=False, load=True):
    """Load stimili and brain measurements.

    :return:
    time_steps: dictonary that contains the steps for each block. dic key is block number.
    brain_activations: {block_number: {time_step: vector of brain activation}
    stimuli: {block_number: {time_step: stimuli representation}
    """
    if load and Path(self.data_dir+"brain_activations.npy").exists():
      brain_activations = np.load(self.data_dir+"brain_activations.npy").item()
      stimuli_in_context = np.load(self.data_dir + "stimuli_in_context.npy").item()
      time_steps = np.load(self.data_dir + "time_steps.npy").item()
      blocks = brain_activations.keys()
    else:
      all_events = self.brain_data_reader.read_all_events(subject_ids=[self.subject_id])
      blocks, time_steps, brain_activations, stimuli_in_context = self.decompose_scan_events(all_events[self.subject_id])

    self.blocks = blocks

    start_steps = {}
    end_steps = {}
    for block in self.blocks:
      start_steps[block] = 0
      while stimuli_in_context[block][start_steps[block]][1] == None:
        start_steps[block] += 1
      end_steps[block] = start_steps[block]
      while stimuli_in_context[block][end_steps[block]][1] != None and end_steps[block] < len(stimuli_in_context[block]):
        end_steps[block] += 1
    print(start_steps)
    print(end_steps)

    if save:
      print("Saving the data ...")
      np.save(self.data_dir + "brain_activations", brain_activations)
      np.save(self.data_dir + "stimuli_in_context", stimuli_in_context)
      np.save(self.data_dir + "time_steps", time_steps)

    return time_steps, brain_activations, stimuli_in_context, start_steps, end_steps

  def decompose_scan_events(self, scan_events):
    """
    :param scan_events:
    :return:
      time_steps: dictonary that contains the steps for each block. dic key is block number.
      brain_activations: {block_number: {time_step: vector of brain activation}
      stimuli: {block_number: {time_step: stimuli representation}
    """

    tokenizer = SpacyTokenizer()


    timesteps = {} # dict(block_id: list)
    brain_activations = {} # dict(block_id: list)
    stimuli_in_context = {} # dict(block_id: list)

    for block in scan_events:
      stimuli_in_context[block.block_id] = []
      brain_activations[block.block_id] = []
      timesteps[block.block_id] = []

      for event in tqdm(block.scan_events):
        # Get the stimuli in context (what are we going to feed to the computational model!)
        context, stimuli_index = block.get_stimuli_in_context(
          scan_event=event,
          tokenizer=tokenizer,
          context_mode=self.hparams.context_mode,
          past_window=self.hparams.past_window,
          future_window=self.hparams.future_window)

        stimuli_in_context[block.block_id].append((context, stimuli_index))
        brain_activations[block.block_id].append(event.scan)
        timesteps[block.block_id].append(event.timestamp)

    return list(stimuli_in_context.keys()), timesteps, brain_activations, stimuli_in_context

  def encode_stimuli(self, stimuli_in_context, integration_fn=np.mean):
    """Applies the text encoder on the stimuli.

    :param stimuli_in_context:
    :return:
    """
    with tf.Session() as sess:
      encoded_stimuli_of_each_block = {}
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      for block in self.blocks:
        encoded_stimuli_of_each_block[block] = []
        print("Encoding stimuli of block:", block)
        contexts, indexes = zip(*stimuli_in_context[block])
        encoded_stimuli = sess.run(self.stimuli_encoder.get_embeddings(contexts, [len(c) for c in contexts], key=self.embedding_type))
        for encoded, index in zip(encoded_stimuli,indexes):
          # TODO(samira): maybe there is a better fix for this?
          if index is None:
            index = [0]
          encoded_stimuli = integration_fn(encoded[index], axis=0)
          encoded_stimuli_of_each_block[block].append(encoded_stimuli)

    return encoded_stimuli_of_each_block

  def metrics(self):
    """
    :return:
      Dictionary of {metric_name: metric_function}
    """
    return {'mean_EV': mean_explain_variance}


  def voxel_preprocess(self, **kwargs):
    """Returns the list of preprocessing functions with their input parameters.

    :param kwargs:
    :return:
    """

    return [(detrend,{'t_r':2.0}), (reduce_mean,{})]

  def eval(self, mapper, brain_activations, encoded_stimuli,
                         test_blocks, time_steps, start_steps, end_steps, test_delay):
    test_encoded_stimuli, test_brain_activations = mapper.prepare_inputs(blocks=test_blocks,
                                                                         timed_targets=brain_activations,
                                                                         sorted_inputs=encoded_stimuli,
                                                                         sorted_timesteps=time_steps,
                                                                         delay=test_delay,
                                                                         start_steps=start_steps,
                                                                         end_steps=end_steps)
    test_brain_activations, _ = self.voxel_selection(test_brain_activations, fit=False)

    self.eval_mapper(mapper, test_encoded_stimuli, test_brain_activations)

  def eval_mapper(self, mapper, encoded_stimuli, brain_activations):
    """Evaluate the mapper based on the defined metrics.

    :param encoded_stimuli:
    :param brain_activations:
    :return:
    """
    mapper_output = mapper.map(inputs=encoded_stimuli,targets=brain_activations)
    predictions = mapper_output['predictions']

    predictions, _ = self.post_train_voxel_selection(predictions)
    brain_activations, _ = self.post_train_voxel_selection(brain_activations)


    for metric_name, metric_fn in self.metrics().items():
      metric_eval = metric_fn(predictions=predictions, targets=brain_activations)
      print(metric_name,":",metric_eval)

  def get_blocks(self):
    """
    :return: blocks of the loaded brain data.
    """
    return self.blocks

  def get_folds(self, fold_index):
    """
    :param fold_index:
    :return:
      train block ids and test block ids for the given fold index.
    """
    if self.folds is None:
      self.folds = get_folds(self.blocks)

    return self.folds[fold_index]

  def voxel_selection(self, brain_activations, fit=False):

    selected_voxels = np.arange(len(brain_activations[0]))
    reduced_brain_activations = brain_activations
    for selector in self.voxel_selectors:
      reduced_brain_activations = selector.select_featurs(reduced_brain_activations, fit)
      selected_voxels = np.asarray(selected_voxels)[selector.get_selected_indexes()]

    return reduced_brain_activations, selected_voxels

  def post_train_voxel_selection(self, brain_activations, predictions=None, labels=None, fit=False):
    """Voxel selection that needs to be applied after training and is based on training results.

    :param brain_activations:
    :param predictions:
    :param labels:
    :param fit:
    :return:
    """
    if fit and labels is not None and predictions is not None:
      print("Fitting post training voxel selectors")
      for selector in self.post_train_voxel_selectors:
        selector.fit(predictions, labels)

    selected_voxels = np.arange(len(brain_activations[0]))
    reduced_brain_activations = brain_activations
    for selector in self.post_train_voxel_selectors:
      reduced_brain_activations = selector.select_featurs(reduced_brain_activations)
      selected_voxels = np.asarray(selected_voxels)[selector.get_selected_indexes()]

    return reduced_brain_activations, selected_voxels

  def preprocess_brain_activations(self, brain_activations, voxel_preprocessings, start_steps, end_steps, resting_norm=False):
    for block in self.blocks:
      for voxel_preprocessing_fn, args in voxel_preprocessings:
        brain_activations[block] = voxel_preprocessing_fn(brain_activations[block], **args)

      if resting_norm:
        brain_activations[block] = minus_average_resting_states(brain_activations[block], brain_activations[block][:start_steps[block]])
    return brain_activations

  def train_mapper(self, brain_activations, encoded_stimuli, train_blocks,
                   test_blocks,time_steps, start_steps, end_steps, train_delay, test_delay, fold_id, save=True):

    mapper = self.mapper_tuple[0](**self.mapper_tuple[1])
    tf.logging.info('Prepare train pairs ...')
    train_encoded_stimuli, train_brain_activations = mapper.prepare_inputs(blocks=train_blocks,
                                                                                timed_targets=brain_activations,
                                                                                sorted_inputs=encoded_stimuli,
                                                                                sorted_timesteps=time_steps,
                                                                                delay=train_delay,
                                                                                start_steps=start_steps,
                                                                                end_steps=end_steps)

    train_brain_activations, selected_voxels = self.voxel_selection(train_brain_activations, fit=True)

    print(train_brain_activations.shape)
    tf.logging.info('Prepare test pairs ...')

    # Train the mapper
    tf.logging.info('Start training ...')
    mapper.train(inputs=train_encoded_stimuli,targets=train_brain_activations)
    tf.logging.info('Training done!')

    # Select voxels based on performance on training set ...
    mapper_output = mapper.map(inputs=train_encoded_stimuli, targets=train_brain_activations)
    predictions = mapper_output['predictions']
    _, post_selected_voxels = self.post_train_voxel_selection(train_brain_activations, predictions=predictions, labels=train_brain_activations, fit=True)


    # Evaluate the mapper
    if eval:
      tf.logging.info('Evaluating ...')
      self.eval_mapper(mapper, train_encoded_stimuli, train_brain_activations)
      if len(test_blocks) > 0:
        self.eval(mapper, brain_activations, encoded_stimuli,
                  test_blocks, time_steps, start_steps, end_steps, test_delay)

    if save:
      print("Saving the model:")
      save_dir = self.model_dir + "_fold_"+str(fold_id)+"_delay_"+str(train_delay)

      # Save the mode
      pickle.dump(mapper, open(save_dir, 'wb'))

      # Save the params
      pickle.dump(self.hparams, open(save_dir+'_params', 'wb'))

      # Save the selected voxels:
      np.save(selected_voxels, open(save_dir+ 'selected_voxels', 'wb'))
      np.save(post_selected_voxels, open(save_dir + '_post_selected_voxels', 'wb'))

    return mapper

  def train_mappers(self, delays=[0], cross_delay=True,eval=True, save=True, fold_index=-1):

    # Add  different options to encode the stimuli (sentence based, word based, whole block based, whole story based)
    # Load the brain data
    tf.logging.info('Loading brain data ...')
    time_steps, brain_activations, stimuli, start_steps, end_steps = self.load_brain_experiment()

    self.blocks = [1]
    tf.logging.info('Blocks: %s' %str(self.blocks))
    print('Example Stimuli %s' % str(stimuli[1][0]))

    # Preprocess brain activations
    brain_activations = self.preprocess_brain_activations(brain_activations,
                                                          voxel_preprocessings=self.voxel_preprocess(),
                                                          start_steps=start_steps, end_steps=end_steps,
                                                          resting_norm=False)

    # Encode the stimuli and get the representations from the computational model.
    tf.logging.info('Encoding the stimuli ...')
    encoded_stimuli = self.encode_stimuli(stimuli)

    # Get the test and training sets
    train_blocks, test_blocks = self.get_folds(fold_index)
    # Pepare the data for the mapping model (testc and train sets)
    print("train blocks:", train_blocks)
    print("test blocks:", test_blocks)

    if cross_delay:
      delay_pairs = itertools.product(delays, repeat=2)
    else:
      delay_pairs = zip(delays,delays)

    trained_mapper_dic = {}
    for train_delay, test_delay in delay_pairs:
      print("Training with train time delay of %d and test time delay of %d" % (train_delay, test_delay))
      if train_delay not in trained_mapper_dic:
        trained_mapper_dic[train_delay] = self.train_mapper(brain_activations, encoded_stimuli,
                                                            train_blocks,test_blocks,
                                                            time_steps, start_steps, end_steps,
                                                            train_delay, test_delay, fold_index, save=save)
      else:
        self.eval(trained_mapper_dic[train_delay], brain_activations, encoded_stimuli,
                         test_blocks, time_steps, start_steps, end_steps, test_delay)





