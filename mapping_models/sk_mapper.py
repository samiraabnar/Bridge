from sklearn.linear_model import *
from mapping_models.basic_mapper import BasicMapper
import numpy as np

class SkMapper(BasicMapper):
  def __init__(self, hparams, model_fn=Ridge):
    super(SkMapper, self).__init__(hparams)
    self.alpha = hparams.alpha
    self.model_fn = model_fn
    self.model = None

  def build(self, is_train=True):
    """Create the model object using model_fn
    """
    self.model = self.model_fn(alpha=self.alpha)

  def map(self, inputs, targets=None):
    if self.model is None:
      self.build()
    predictions = self.model.predict(inputs)

    loss = None
    if targets is not None:
      loss = self.compute_loss(predictions, targets)

    return {'predictions': predictions,
            'loss': loss}

  def train(self, inputs, targets):
    if self.model is None:
      self.build()

    self.model.fit(inputs, targets)

  def prepare_inputs(self, **kwargs):
    blocks = kwargs['blocks']
    timed_targets = kwargs['timed_targets']
    timed_inputs =  kwargs['sorted_inputs']
    time_steps =  kwargs['sorted_timesteps']

    delay = kwargs['delay']

    inputs = []
    targets = []
    for block in blocks:
      # for all steps in the current block
      for step in time_steps[block]:
        if step+delay in time_steps[block]:
          input_index = np.where(time_steps[block] == step+delay)[0]
          if len(timed_inputs[block][input_index]) > 0:
            inputs.append(timed_inputs[block][input_index][0])
            targets.append(timed_targets[block][step])

    print(np.asarray(targets).shape)
    return np.asarray(inputs), np.asarray(targets)



