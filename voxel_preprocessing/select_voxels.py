from sklearn.feature_selection import VarianceThreshold


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

