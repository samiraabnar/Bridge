"""Pipeline for predicting brain activations.
"""

from evaluation.metrics import  mean_explain_variance
from util.misc import get_folds
import tensorflow as tf
import numpy as np

class ExplainBrain(object):
  def __init__(self, brain_data_reader, stimuli_encoder, mapper):
    self.brain_data_reader = brain_data_reader
    self.stimuli_encoder = stimuli_encoder
    self.mapper = mapper
    self.blocks = None
    self.folds = None
    self.subject_id = 1

  def load_brain_experiment(self):
    """Load stimili and brain measurements.

    :return:
    time_steps: dictonary that contains the steps for each block. dic key is block number.
    brain_activations: {block_number: {time_step: vector of brain activation}
    stimuli: {block_number: {time_step: stimuli representation}
    """

    all_events = self.brain_data_reader.read_all_events(subject_ids=[self.subject_id])
    blocks, time_steps, brain_activations, stimuli = self.decompose_scan_events(all_events[self.subject_id])

    self.blocks = blocks
    return time_steps, brain_activations, stimuli

  def decompose_scan_events(self, scan_events):
    """
    :param scan_events:
    :return:
      time_steps: dictonary that contains the steps for each block. dic key is block number.
      brain_activations: {block_number: {time_step: vector of brain activation}
      stimuli: {block_number: {time_step: stimuli representation}
    """
    time_steps = {}
    brain_activations = {}
    stimuli = {}

    for block in scan_events:
      block = block
      block_events = scan_events[block]
      if block not in brain_activations:
        brain_activations[block] = {}
        stimuli[block] = []
        time_steps[block] = []
        for event in block_events:
          time_steps[block].append(event.timestamp)
          stimuli[block].append(event.stimulus)
          brain_activations[block][event.timestamp] = event.scan

    blocks = list(brain_activations.keys())

    sorted_timesteps = {}
    sorted_stimuli = {}
    for block in blocks:
      sorted_indexes = np.argsort(time_steps[block])
      sorted_timesteps[block] = np.asarray(time_steps[block])[sorted_indexes]
      sorted_stimuli[block] = np.asarray(stimuli[block])[sorted_indexes]

    return blocks, sorted_timesteps, brain_activations, sorted_stimuli


  def encode_stimuli(self, stimuli, integration_fn=np.mean):
    """Applies the text encoder on the stimuli.

    :param stimuli:
    :return:
    """
    with tf.Session() as sess:
      encoded_stimuli_of_each_block = {}
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      for block in self.blocks:
        print("Encoding stimuli of block:", block)
        print(stimuli[block])
        encoded_stimuli_of_each_block[block] = integration_fn(
          sess.run(self.stimuli_encoder.get_embeddings(stimuli[block])),
          axis=1)

    print(encoded_stimuli_of_each_block[1].shape)
    return encoded_stimuli_of_each_block

  def metrics(self):
    """
    :return:
      Dictionary of {metric_name: metric_function}
    """
    return {'mean_EV': mean_explain_variance}

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



  def train_mapper(self, delay=0, eval=True, save=True, fold_index=-1):

    # Add  different option to encode the stimuli (sentence based, word based, whole block based, whole story based)
    # Load the brain data
    tf.logging.info('Loading brain data ...')
    time_steps, brain_activations, stimuli = self.load_brain_experiment()
    tf.logging.info('Blocks: %s' %str(self.blocks))
    print('Example Stimuli %s' % str(stimuli[1][0]))

    # Encode the stimuli and get the representations from the computational model.
    tf.logging.info('Encoding the stimuli ...')
    encoded_stimuli = self.encode_stimuli(stimuli)

    # Get the test and training sets
    train_blocks, test_blocks = self.get_folds(fold_index)
    # Pepare the data for the mapping model (test and train sets)
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




