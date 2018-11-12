from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn import preprocessing
import numpy as np
import itertools
from util.misc import generate_pair_for_comparison

def mse(predictions, targets):
  """Mean Squared Error.

  :param predictions: (n_samples, n_outputs)
  :param targets: (n_samples, n_outputs)
  :return:
    a scalar which is mean squared error
  """
  return  mean_squared_error(predictions, targets)

def explained_variance(predictions, targets):
  return explained_variance_score(targets, predictions, multioutput='raw_values')

def mean_explain_variance(predictions, targets):
  return explained_variance_score(targets, predictions, multioutput='uniform_average')

def cosine_similarity(predictions, targets):
  raise NotImplementedError()

def binary_accuracy(predictions, targets):
  raise NotImplementedError()

def binary_accuracy_from_dists(cosine_dists, euclidean_dists):
  nn_index = np.argmin(cosine_dists, axis=1)
  accuracy_on_test = np.mean(nn_index == np.argmax(np.eye(cosine_dists.shape[0]), axis=1))

  b_acc = []
  e_b_acc = []
  for i, j in itertools.combinations(np.arange(cosine_dists.shape[0]), 2):
    right_match = cosine_dists[i, i] + cosine_dists[j, j]
    wrong_match = cosine_dists[i, j] + cosine_dists[j, i]
    b_acc.append(right_match < wrong_match)

    e_right_match = euclidean_dists[i, i] + euclidean_dists[j, j]
    e_wrong_match = euclidean_dists[i, j] + euclidean_dists[j, i]
    e_b_acc.append(e_right_match < e_wrong_match)

  return np.mean(b_acc), np.mean(e_b_acc), b_acc, e_b_acc

def MRR(distances):
  prec_at_corrects = []
  ranks = []
  sorted_indexes = np.argsort(distances, axis=1)
  for i in np.arange(len(distances)):
    # print(i)
    correct_at = np.where(sorted_indexes[i] == i)[0] + 1
    # print("Reciprocal Rank",correct_at)
    prec_at_correct = 1.0 / correct_at
    # print("precision at ",correct_at,": ",prec_at_correct)
    prec_at_corrects.append(prec_at_correct)
    ranks.append(correct_at)

  print("MRR: ", np.mean(prec_at_corrects), " ", np.mean(ranks))
  return np.mean(ranks), np.mean(prec_at_corrects), ranks, prec_at_corrects

  def pairwise_cosine_simple(gold_data, prediction, random_data):
    similarity_with_prediction = cosine_similarity(prediction, gold_data)
    random_similarity = cosine_similarity(prediction, random_data)
    return similarity_with_prediction >= random_similarity


def pairwise_cosine_complex(data, gold_prediction, random_prediction):
  similarity_with_prediction = cosine_similarity(gold_prediction, data)
  random_similarity = cosine_similarity(random_prediction, data)
  return similarity_with_prediction >= random_similarity


def pairwise_euclidean_simple(scan, representation, random_scan):
  distance_to_true_scan = np.linalg.norm(preprocessing.normalize(representation) - preprocessing.normalize(scan))
  distance_to_random_scan = np.linalg.norm(
    preprocessing.normalize(representation) - preprocessing.normalize(random_scan))
  return distance_to_true_scan <= distance_to_random_scan


def pairwise_euclidean_complex(scan, representation, random_prediction):
  distance_to_true_scan = np.linalg.norm(preprocessing.normalize(representation) - preprocessing.normalize(scan))
  distance_to_random_scan = np.linalg.norm(preprocessing.normalize(random_prediction) - preprocessing.normalize(scan))
  return distance_to_true_scan <= distance_to_random_scan


# Return score for comparison, either 0 or 1
def compare_pairs(data, predict, random, config, method):
  score = 0
  if (config == "simple"):
    if (pairwise_euclidean_simple(data, predict, random)):
      score += 1
  elif (pairwise_euclidean_complex(data, predict, random)):
    score += 1
  return score


# For n classifications, generate pairs of certain sequence lenght for comparison and add 1 if classification was correct 0 otherwise.
def classification_task(brainData, predictions, sequenceLength, classifications=10000, config="simple",
                        method="euclidean"):
  n = classifications
  result = 0
  while (n > 0):
    brainData, predictedData, randomComparison = generate_pair_for_comparison(brainData, predictions, sequenceLength,
                                                                              config)
    result += compare_pairs(brainData, predictedData, randomComparison, config, method)
    n -= 1
  return result / classifications