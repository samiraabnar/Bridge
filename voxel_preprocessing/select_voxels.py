from sklearn.feature_selection import VarianceThreshold
from evaluation.metrics import explained_variance
import numpy as np

class FeatureSelection(object):
  def select_featurs(self, data):
    raise NotImplementedError()


class VarianceFeatureSelection(object):
  def __init__(self, variance_threshold=0.0):
    self.model = VarianceThreshold((0.0))

  def fit(self, timeseries_data):
    self.model.fit(timeseries_data)

  def select_featurs(self, timeseries_data, fit=False):
    if fit:
      self.model.fit(timeseries_data)

    return self.model.transform(timeseries_data)


  def get_selected_indexes(self):
    return self.model.get_support(indices=False)



class TopkFeatureSelection(object):
  def __init__(self, metric_fn=explained_variance, k=1000):
    self.metric_fn = metric_fn
    self.selected_indexes = None
    self.k = k

  def fit(self, data_predictions, data_labels):
    metric_eval = self.metric_fn(data_predictions, data_labels)
    self.selected_indexes = np.argsort(metric_eval)[-self.k:]

  def select_featurs(self, data):
    return data[self.selected_indexes]


  def get_selected_indexes(self):
    return self.selected_indexes


if __name__ == '__main__':
    fs = TopkFeatureSelection()