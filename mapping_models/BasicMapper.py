from evaluation.metrics import  mse

class BasicMapper(object):
  def __init__(self, hparams):
    self.hparams = hparams

  def build(self, is_train):
    raise NotImplementedError

  def map(self, inputs, targets):
    raise NotImplementedError

  def prepare_inputs(self, **kwargs):
    raise NotImplementedError()

  def compute_loss(self, predictions, targets):
    return mse(predictions, targets)


