from read_dataset import readHarryPotterData

from computational_model import combine_representations, mapping, load_representations
from language_preprocessing import tokenize
from evaluation import evaluate
import spacy
import numpy as np
import pickle

harry_dir = "/Users/lisa/Corpora/HarryPotter/"

# READ
print("\n\nHarry Potter Data")
# First filter data to contain only one subject. Once it is set up, we will probably have a loop over all subjects
harry_data = [e for e in readHarryPotterData.read_all(harry_dir) if e.subject_id == "1"]
#
# # TOKENIZE: This takes way too long!
harry_data = tokenize.tokenize_all(harry_data, spacy.load('en_core_web_sm'))
#
# # Tokenization takes too long, so we save an interim format to save time in later runs
with open('harrydata_subj1.pickle', 'wb') as handle:
     pickle.dump(harry_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

#with open('harrydata_subj1.pickle', 'rb') as handle:
#    harry_data = pickle.load(handle)
block1 = [e for e in harry_data if e.block == 1]
block2 = [e for e in harry_data if e.block == 2]
block3 = [e for e in harry_data if e.block == 3]
block4 = [e for e in harry_data if (e.subject_id == "1" and e.block == 4)]

scan_events = [block1, block2, block3, block4]

# The last scan of each block contains all sentences seen in this block.
sentences = [block1[-1].sentences, block2[-1].sentences, block3[-1].sentences, block4[-1].sentences]

# scans
block1_scans = [event.scan for event in block1]
block2_scans = [event.scan for event in block2]
block3_scans = [event.scan for event in block3]
block4_scans = [event.scan for event in block4]
scans = [block1_scans, block2_scans, block3_scans, block4_scans]

# Embedding takes long, so we save an interim format to save time in later runs
embeddings = [load_representations.elmo_embed(sents) for sents in sentences]
with open('harrydata_embeddings.pickle', 'wb') as handle:
    pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

#with open('harrydata_embeddings.pickle', 'rb') as handle:
 #   embeddings = pickle.load(handle)


stimulus_embeddings = []

# Get embeddings for each stimulus in each block
for block in range(0,4):
    block_stimulus_embeddings = combine_representations.get_stimulus_embeddings(sentences[block],embeddings[block], scans[block])
    stimulus_embeddings.append(block_stimulus_embeddings)


# Prepare train and test data

train_scans =[block1_scans, block2_scans, block3_scans]
train_embeddings = stimulus_embeddings[0:2]
test_scans = [block4_scans]
test_embeddings = stimulus_embeddings[3]


# Set delay to 2 timesteps, for the Harry Data this equals to 4 seconds
# VERY naive implementation of delay! Adjust!
delay = 2
train_scans, train_embeddings = combine_representations.add_delay(delay, train_scans, train_embeddings)
test_scans, test_embeddings = combine_representations.add_delay(delay, test_scans, test_embeddings)


#  learn mapping from embeddings to activations
mapping_model = mapping.learn_mapping(train_embeddings, train_scans)
#  apply mapping on test data
predicted_scans = mapping.predict(mapping_model, stimulus_embeddings[3])

#  Evaluate
print("Explained variance for block 4, subject1")
print(evaluate.voxelwise_explained_variance(block4_scans, predicted_scans))
