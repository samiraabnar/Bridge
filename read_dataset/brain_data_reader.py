"""Basic class for Brain Readers.

"""


class BrainDataReader(object):
  def __init__(self, data_dir):
    self.data_dir = data_dir

  def read_all_events(self, subject_id):
    raise NotImplementedError()


class FmriReader(object):
  def __init__(self, data_dir):
    self.data_dir = data_dir

  def read_all_events(self, **kwargs):
    raise NotImplementedError()