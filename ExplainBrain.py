"""Pipeline for predicting brain activations.
"""

from evaluation.metrics import  mean_explain_variance
from util.misc import get_folds
from voxel_preprocessing.preprocess_voxels import detrend
from voxel_preprocessing.preprocess_voxels import minus_average_resting_states
from language_preprocessing.tokenize import SpacyTokenizer

import tensorflow as tf
import numpy as np
from tqdm import tqdm


class ExplainBrain(object):
  def __init__(self, brain_data_reader, stimuli_encoder, mapper):
    self.brain_data_reader = brain_data_reader
    self.stimuli_encoder = stimuli_encoder
    self.mapper = mapper
    self.blocks = None
    self.folds = None
    self.subject_id = 1

    self.data_dir = "../bridge_data/processed/harrypotter/"+str(self.subject_id)+"_"

  def load_brain_experiment(self, voxel_preprocessings=[], save=False, load=True):
    """Load stimili and brain measurements.

    :return:
    time_steps: dictonary that contains the steps for each block. dic key is block number.
    brain_activations: {block_number: {time_step: vector of brain activation}
    stimuli: {block_number: {time_step: stimuli representation}
    """
    if load:
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
      np.save(self.data_dir+"brain_activations", brain_activations)
      np.save(self.data_dir + "stimuli_in_context", stimuli_in_context)
      np.save(self.data_dir + "time_steps", time_steps)

    return time_steps, brain_activations, stimuli_in_context, start_steps, end_steps

  def decompose_scan_events(self, scan_events, context_mode='sentence'):
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
          context_mode='sentence',
          past_window=0)

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
        encoded_stimuli = sess.run(self.stimuli_encoder.get_embeddings(contexts, [len(c) for c in contexts]))
        for encoded, index in zip(encoded_stimuli,indexes):
          encoded_stimuli = integration_fn(encoded[index], axis=-1)
          encoded_stimuli_of_each_block[block].append(encoded_stimuli)

    return encoded_stimuli_of_each_block

  def metrics(self):
    """
    :return:
      Dictionary of {metric_name: metric_function}
    """
    return {'mean_EV': mean_explain_variance}


  def voxel_preprocess(self, **kwargs):

    return [(detrend,{'t_r':2.0})]

  def eval_mapper(self, encoded_stimuli, brain_activations):
    """Evaluate the mapper based on the defined metrics.

    :param encoded_stimuli:
    :param brain_activations:
    :return:
    """
    mapper_output = self.mapper.map(inputs=encoded_stimuli,targets=brain_activations)

    predictions = mapper_output['predictions']
    print(predictions.shape)
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

  def preprocess_brain_activations(self, brain_activations, voxel_preprocessings, start_steps, end_steps):
    for block in brain_activations.keys():
      start_step = []
      for voxel_preprocessing_fn, args in voxel_preprocessings:
        brain_activations[block] = voxel_preprocessing_fn(brain_activations[block], **args)

    return brain_activations

  def train_mapper(self, delay=0, eval=True, save=True, fold_index=-1):

    # Add  different option to encode the stimuli (sentence based, word based, whole block based, whole story based)
    # Load the brain data
    tf.logging.info('Loading brain data ...')
    time_steps, brain_activations, stimuli, start_steps, end_steps = self.load_brain_experiment()
    tf.logging.info('Blocks: %s' %str(self.blocks))
    print('Example Stimuli %s' % str(stimuli[1][0]))

    brain_activations = self.preprocess_brain_activations(brain_activations, voxel_preprocessings=self.voxel_preprocess())

    # Encode the stimuli and get the representations from the computational model.
    tf.logging.info('Encoding the stimuli ...')
    encoded_stimuli = self.encode_stimuli(stimuli, start_steps, end_steps )

    # Get the test and training sets
    train_blocks, test_blocks = self.get_folds(fold_index)
    # Pepare the data for the mapping model (testc and train sets)
    print("train blocks:", train_blocks)
    print("test blocks:", test_blocks)

    tf.logging.info('Prepare train pairs ...')

    train_encoded_stimuli, train_brain_activations = self.mapper.prepare_inputs(blocks=train_blocks,
                                                                                timed_targets=brain_activations,
                                                                                sorted_inputs=encoded_stimuli,
                                                                                sorted_timesteps=time_steps,
                                                                                delay=delay)

    tf.logging.info('Prepare test pairs ...')
    if len(test_blocks) > 0:
      test_encoded_stimuli, test_brain_activations = self.mapper.prepare_inputs(blocks=test_blocks,
                                                                                timed_targets=brain_activations,
                                                                                sorted_inputs=encoded_stimuli,
                                                                                sorted_timesteps=time_steps,
                                                                                delay=delay)
    else:
        print('No test blocks!')

    # Train the mapper
    tf.logging.info('Start training ...')
    self.mapper.train(inputs=train_encoded_stimuli,targets=train_brain_activations)
    tf.logging.info('Training done!')

    # Evaluate the mapper
    if eval:
      tf.logging.info('Evaluating ...')
      self.eval_mapper(train_encoded_stimuli, train_brain_activations)
      if len(test_blocks) > 0:
        self.eval_mapper(test_encoded_stimuli, test_brain_activations)




