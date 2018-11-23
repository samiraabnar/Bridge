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

### files:
Path to the raw brain data, e.g.: '/Users/samiraabnar/Codes/Data/harrypotter/'

Spacy model file for tokenization:
python -m spacy download en_core_web_lg

## How to run:
python run_experiment.py
