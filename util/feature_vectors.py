from sklearn.feature_selection import VarianceThreshold


def intersection(list_of_lists):
  intersection_list = []
  for list in list_of_lists:
    if len(intersection_list) == 0:
      intersection_list = list(set(list))
    else:
      intersection_list = list(intersection_list & set(list))

  return intersection_list

class FeatureSelector(object):
  def __init__(self, feature_selector=VarianceThreshold(threshold=0.0)):
    """

    :param feature_selector: e.g. PCA(n_components=512), VarianceThreshold(threshold=0.0)
    """
    self.feature_selector = feature_selector

  def fit(self,feature_vectors):
    self.feature_selector.fit(feature_vectors)

  def fit_transform(self,feature_vectors):
    return self.feature_selector.fit_transform(feature_vectors)

  def transform(self, feature_vectors):
    return self.feature_selector.transform(feature_vectors)

  def get_selected_indices(self):
    return self.feature_selector.get_support(indices=True)
