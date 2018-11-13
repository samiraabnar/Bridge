"""Random usefull functions.

"""

from itertools import combinations
import numpy as np
import random

def get_folds(block_list, test_ratio=0.25):
  """Given the list of blocks splits them into all possible test and train sets

  :param block_list:
  :param test_ratio:
  :return:
  """
  total_number_of_blocks = len(block_list)
  test_size = int(total_number_of_blocks * test_ratio)
  train_size = total_number_of_blocks - test_size
  train_blocks = list([list(a) for a in combinations(block_list,train_size)])

  test_blocks = []
  for i in np.arange(len(train_blocks)):
    test_block = [x for x in block_list if x not in train_blocks[i]]
    test_blocks.append(test_block)


  return list(zip(train_blocks, test_blocks))


# Randomly select brain and predicted data for comparison of sequenceLength n: e.g. 4, 8 or 12 words.
def generate_pair_for_comparison(brainData, predictions, sequenceLength, config):
  maximum = len(brainData) - 1 - (sequenceLength - 1)
  # Select random sequence of gold data and corresponding predictions
  start = random.randint(0, maximum)
  brainScans = brainData[start:start + sequenceLength]
  predictedScans = predictions[start:start + sequenceLength]
  # Select random sequence for comparison
  # Simple task: compare one prediction against random and correct brain data
  # Complex task: compare brain data against correct and random prediction
  randomStart = random.randint(0, maximum)
  if (config == "simple"):
    randomComparison = brainData[randomStart:randomStart + sequenceLength]
  elif (config == "complex"):
    randomComparison = predictions[randomStart:randomStart + sequenceLength]
  return brainScans, predictedScans, randomComparison


def pad_lists(list_of_list, padding=''):
  padded_lists = []
  max_length = np.max(list(map(lambda x: len(x), list_of_list)))

  for l in list_of_list:
    padded_lists.append(l + (max_length - len(l))*[padding])

  return padded_lists


if __name__ == '__main__':
    print(pad_lists([['a','b'],['c']], ''))