"""Objects for storing fMRI events.

"""
import numpy as np

class ScanEvent(object):
  def __init__(self,subject_id, stimulus_pointer, timestamp, scan):
    """

    :param subject_id:
    :param stimulus_pointer: List of tuples (sentence_id, token_id)
    :param timestamp:
    :param scan: Array of voxel activations
    """
    self.subject_id = subject_id
    self.stimulus_pointer = stimulus_pointer
    self.timestamp = timestamp
    self.scan = scan


class Block(object):
  """One experimental block or run.
  """

  def __init__(self, subject_id, block_id, sentences=[], scan_events=[], voxel_to_region_mapping=None):
    """

    :param block_id:
    :param subject_id:
    :param sentences: List of sentences presented during the block, each sentence is a list of tokens.
                      Tokens are stored as presented to the participants, no preprocessing in the reader.
                      [["Tokens", "in", "the", " first", "sentence."], ["And", "tokens", "in", "sentence", "number", "two."]]
    :param scan_events: List of scan events occurring in this block
    :param voxel_to_region_mapping: If available: a mapping from the nth-voxel in each scan to the corresponding brain region
    """
    self.block_id = block_id
    self.subject_id = subject_id
    self.sentences = sentences
    self.scan_events = scan_events
    self.voxel_to_region_mapping = voxel_to_region_mapping


  def get_sentence_context(self, scan_event, tokenizer, only_past=False, past_window=0, future_window=0):
    """

    :param scan_event:
    :param tokenizer: object with a tokenize function that gets a string and returns a string.
    :param past_window:
    :param future_window:
    :return:
      context (list of tokens), indexes of stimuli in the context
    """
    if len(scan_event.stimulus_pointer) > 0:
      if only_past:
        future_window = 0

      sentence_indexes, token_indexes = list(zip(*scan_event.stimulus_pointer))
      current_sentence_index_start = scan_event.stimulus_pointer[0][0]
      current_sentence_index_end = scan_event.stimulus_pointer[-1][0]
      start_sentence_index = np.max([current_sentence_index_start - past_window, 0])
      last_sentence_index = np.min([len(self.sentences), current_sentence_index_end + future_window])

      stimuli_indexes = []
      context = []
      for i in np.arange(start_sentence_index, last_sentence_index+1):

        # Compute the index of stimuli in the context
        stimuli_tokens_in_current_sent_indexes = np.where(sentence_indexes == i)[0]
        if len(stimuli_tokens_in_current_sent_indexes) > 0:
          current_sentence = self.sentences[i]
          for ind in np.arange(0,len(current_sentence)):
            context_len_before = len(context)
            context.extend(tokenizer.tokenize(self.sentences[i][ind]))
            if ind in list(np.asarray(token_indexes)[stimuli_tokens_in_current_sent_indexes]):
              stimuli_indexes.extend(np.arange(context_len_before, len(context)))
        else:
          context.extend(tokenizer.tokenize(self.sentences[i]))


      return context, stimuli_indexes
    else:
      return [''], None



  def get_stimuli_in_context(self,scan_event, tokenizer, context_mode, **kwargs):
    """

    :param scan_event:
    :param tokenizer:
    :param context_mode:
    :param kwargs:
    :return:
      context (list of tokens), indexes of stimuli in the context
    """
    if context_mode == 'sentence':
      return self.get_sentence_context(scan_event=scan_event, tokenizer=tokenizer,
                                  **kwargs)
    else:
      raise NotImplementedError()
