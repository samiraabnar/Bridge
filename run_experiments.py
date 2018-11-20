"""Main file to run for training and evaluating the models.

"""
from ExplainBrain import ExplainBrain
from read_dataset.harrypotter_data import HarryPotterReader
from computational_model.text_encoder import TfHubElmoEncoder
from mapping_models.sk_mapper import SkMapper
import tensorflow as tf


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('subject_id', 1, 'subject id')
tf.flags.DEFINE_integer('fold_id', 0, 'fold id')
tf.flags.DEFINE_list('delays',[-8,-6,-4,-2,0,2,4,6,8] , 'delay list')
tf.flags.DEFINE_boolean('cross_delay', True, 'try different train and test delays')

tf.flags.DEFINE_float('alpha', 1.0, 'alpha')
tf.flags.DEFINE_string('embedding_dir', None, 'path to the file containing the embeddings')
tf.flags.DEFINE_string('brain_data_dir', '/Users/samiraabnar/Codes/Data/harrypotter/', 'Brain Data Dir')

tf.flags.DEFINE_string('embedding_type', 'word_emb', 'type of embedding')
tf.flags.DEFINE_string('context_mode', 'none', 'type of context (sentence, block, none)')
tf.flags.DEFINE_integer('past_window', 0, 'window size to the past')
tf.flags.DEFINE_integer('future_window', 0, 'window size to the future')

tf.flags.DEFINE_boolean('save_data', True ,'save data flag')
tf.flags.DEFINE_boolean('load_data', True ,'load data flag')

tf.flags.DEFINE_boolean('save_models', True ,'save models flag')

hparams = FLAGS
if __name__ == '__main__':

  tf.logging.set_verbosity(tf.logging.INFO)

  # Define how we want to read the brain data
  print("1. initialize brain data reader ...")
  brain_data_reader = HarryPotterReader(data_dir=hparams.brain_data_dir)

  # Define how we want to computationaly represent the stimuli
  print("2. initialize text encoder ...")
  stimuli_encoder = TfHubElmoEncoder(hparams)

  print("3. initialize mapper ...")
  mapper = (SkMapper, {'hparams':hparams})

  # Build the pip
  # eline object
  print("4. initialize Explainer...")
  explain_brain = ExplainBrain(hparams, brain_data_reader, stimuli_encoder, mapper)

  # Train and evaluate how well we can predict the brain activatiobs
  print("5. train and evaluate...")
  explain_brain.train_mappers(delays=hparams.delays, cross_delay=hparams.cross_delay,eval=True, fold_index=hparams.fold_id)