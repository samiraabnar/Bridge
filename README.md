# Bridge
Making a bridge between NLP models and Brain data

This repository includes codes for:
* Using internal states of a pretrained language model to predict brain activations


## Requirements:
* tensorflow
* tensorflow_hub
* numpy
* sklearn
* spacy
* https://github.com/samiraabnar/GoogleLM1b.git
* https://github.com/google-research/bert

### files:
Path to the raw brain data, e.g.: '/Users/samiraabnar/Codes/Data/harrypotter/'

Spacy model file for tokenization:
python -m spacy download en_core_web_lg

## How to obtain contextualized representations from different models:
python encode_stimuli_in_context.py
## How to run predictive modelling experiments:
python run_experiment.py

## How to run RSA experiments:
python rsa/compute_rep_sim.py

## Link to colab for plotting the results:
https://colab.research.google.com/drive/1hO9ZV6gfO-WT-wGFwJZvHPQrLPHHuuhO#scrollTo=HeHmO4ZTL9O5&uniqifier=10


**To create story_features.npy you can use the scripts in this colab notebook**
