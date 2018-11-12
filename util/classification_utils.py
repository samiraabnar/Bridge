import numpy as np

def prepare_input_output_pairs(train_block_ids, test_block_ids, block_scans, block_features, mode="concat"):
  train_brain_activations = {}
  train_features = {}
  test_brain_activations = {}
  test_features = {}

  for subject_id in np.arange(1, 9):
    train_features[subject_id] = {}
    train_brain_activations[subject_id] = []
    for feautre_key in block_features.keys():
      train_features[subject_id][feautre_key] = []

    for train_block_id in train_block_ids:
      for i in np.arange(len(block_scans[subject_id][train_block_id])):
        scan = block_scans[subject_id][train_block_id][i]
        train_brain_activations[subject_id].append(scan.activations[0])

        step = scan.step - block_scans[subject_id][train_block_id][0].step

        for feautre_key in block_features.keys():
          i = 0
          feature_values = []

          while i < 4 and (step + i) < len(block_features[feautre_key][train_block_id]):
            feature_values.append(block_features[feautre_key][train_block_id][step + i])
            i += 1
          while len(feature_values) < 4:
            feature_values.append(np.zeros_like(feature_values[-1]))

          if mode == "concat":
            if len(np.asarray(feature_values).shape) > 1:
              train_features[subject_id][feautre_key].append(np.concatenate(np.asarray(feature_values)))
            else:
              train_features[subject_id][feautre_key].append(np.asarray(feature_values))

          else:
            train_features[subject_id][feautre_key].append(np.mean(feature_values, axis=0))

    test_brain_activations[subject_id] = []
    test_features[subject_id] = {}
    for feautre_key in block_features.keys():
      test_features[subject_id][feautre_key] = []

    for test_block_id in test_block_ids:
      for i in np.arange(len(block_scans[subject_id][test_block_id])):
        scan = block_scans[subject_id][test_block_id][i]
        test_brain_activations[subject_id].append(scan.activations[0])

        step = scan.step - block_scans[subject_id][test_block_id][0].step
        for feautre_key in block_features.keys():
          i = 0
          feature_values = []
          while i < 4 and (step + i) < len(block_features[feautre_key][test_block_id]):
            feature_values.append(block_features[feautre_key][test_block_id][step + i])
            i += 1

          while len(feature_values) < 4:
            feature_values.append(np.zeros_like(feature_values[-1]))

          if mode == "concat":
            if len(np.asarray(feature_values).shape) > 1:
              test_features[subject_id][feautre_key].append(np.concatenate(feature_values))
            else:
              test_features[subject_id][feautre_key].append(feature_values)
          else:
            test_features[subject_id][feautre_key].append(np.mean(feature_values, axis=0))

  return train_features, train_brain_activations, test_features, test_brain_activations
