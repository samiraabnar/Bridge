# In: List of voxel vectors, list of stimuli vectors
# Out: Score for quality of stimuli vectors
# Challenge: How can we best evaluate representations for continuous language?

# Parameters:
# evaluation measure: 2x2, accuracy, explained variance, ...?
# setup: cross-validation setting, train-test split etc
# hemodynamic delay
# mode of representation combination (concatenation, average etc)
# --> maybe the combination mode could also be decided earlier when loading representations

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import explained_variance_score
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import mean_squared_error



# The following evaluations operate on a single (scan, prediction) pair
# Evaluate if the correlation between the predicted representation and the gold scan is higher
# than the correlation between the predicted representation and random_data.
# The random instance can be either a randomly selected scan from another time step OR
# A prediction for a randomly selected stimulus from another time step

# This is the evaluation method used by Mitchell et al. (2008) WHICH ONE?


def pairwise_cosine(gold_scan, prediction, random_instance):
    similarity_with_prediction = cosine_similarity(prediction, gold_scan)
    random_similarity = cosine_similarity(prediction, random_instance)
    return similarity_with_prediction >= random_similarity


def pairwise_euclidean(gold_scan, prediction, random_instance):
    distance_to_gold_scan = np.linalg.norm(preprocessing.normalize(prediction) - preprocessing.normalize(gold_scan))
    distance_to_random_scan = np.linalg.norm(
        preprocessing.normalize(prediction) - preprocessing.normalize(random_instance))
    return distance_to_gold_scan <= distance_to_random_scan


# The following evaluation metrics operate on all predictions.
# Gold_data and predictions are matrices of the same size with the following structure
# Rows correspond to samples (scans at different time steps), columns correspond to voxels

def distance_based_accuracy(gold_scans, predictions, random_data, metric="cosine"):
    if not len(gold_scans) == predictions and len(predictions) == random_data:
        raise ValueError("The number of gold scans, predictions and random data is not equal.")
    results = []
    for i in range(len(gold_scans)):
        if metric == "euclidean":
            results.append(pairwise_euclidean(gold_scans[i], predictions[i], random_data[i]))
        else:
            results.append(pairwise_cosine(gold_scans[i], predictions[i], random_data[i]))
    return sum(results) / float(len(results))


# def canonical_correlation_analysis():
# TODO CCA

# def representational_similarity_analysis:
# TODO RSA

# The following methods are redundant, because they do not do anything that sklearn does not already do!


def voxelwise_explained_variance(gold_data, predictions, multioutput='uniform_average'):
    return explained_variance_score(gold_data, predictions, multioutput)


def mean_squared_error(gold_data, predictions, multioutput='uniform_average'):
    return mean_squared_error(gold_data, predictions, multioutput)

#
#
# # What are you calculating here?
# def binary_accuracy_from_dists(cosine_dists, euclidean_dists):
#     nn_index = np.argmin(cosine_dists, axis=1)
#     accuracy_on_test = np.mean(nn_index == np.argmax(np.eye(cosine_dists.shape[0]), axis=1))
#
#     b_acc = []
#     e_b_acc = []
#     for i, j in itertools.combinations(np.arange(cosine_dists.shape[0]), 2):
#         right_match = cosine_dists[i, i] + cosine_dists[j, j]
#         wrong_match = cosine_dists[i, j] + cosine_dists[j, i]
#         b_acc.append(right_match < wrong_match)
#
#         e_right_match = euclidean_dists[i, i] + euclidean_dists[j, j]
#         e_wrong_match = euclidean_dists[i, j] + euclidean_dists[j, i]
#         e_b_acc.append(e_right_match < e_wrong_match)
#
#     return np.mean(b_acc), np.mean(e_b_acc), b_acc, e_b_acc
#
# # I do not yet understand what this is for
# def MRR(distances):
#     prec_at_corrects = []
#     ranks = []
#     sorted_indexes = np.argsort(distances, axis=1)
#     for i in np.arange(len(distances)):
#         # print(i)
#         correct_at = np.where(sorted_indexes[i] == i)[0] + 1
#         # print("Reciprocal Rank",correct_at)
#         prec_at_correct = 1.0 / correct_at
#         # print("precision at ",correct_at,": ",prec_at_correct)
#         prec_at_corrects.append(prec_at_correct)
#         ranks.append(correct_at)
#
#     print("MRR: ", np.mean(prec_at_corrects), " ", np.mean(ranks))
#     return np.mean(ranks), np.mean(prec_at_corrects), ranks, prec_at_corrects
