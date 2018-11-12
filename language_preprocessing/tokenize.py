# This method is horribly inefficient because the content of sentences is incremented over the scans and we tokenize it again and again!
# Would probably be smarter to do this in the reader directly

# The tokenization code so far only works if the model is initialized as
# model = spacy.load('en_core_web_sm')
# Would probably have to refactor a lot if I want to change the tokenization

<<<<<<< HEAD
import re
import spacy


def naive_word_level_tokenizer(text):
  """

  :param text: a string
  :return:
    splitted text
  """
  splitted_text = re.split("([A-Za-z\"]+)", text)
  return splitted_text


def stimuli_tokenizer(stimuli_sequence, stimuli_steps, tokenize_fn):
  """

  :param stimuli_sequence:
  :param stimuli_steps:
  :param tokenize_fn:
  :return:
    tokens, steps
  """
  tokens = []
  steps = []

  for stimuli, step in  zip(stimuli_sequence, stimuli_steps):
    tokenized_stimuli = tokenize_fn(stimuli)
    for tok in tokenized_stimuli:
      tokens.append(tok)
      steps.append(step)

  return tokens, steps


# I am using spacy tokenization here, because this is what seems to be used by Elmo
def spacy_tokenize(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        sentences.append([tok.text for tok in sent])
    return sentences

=======
def tokenize_all(scan_events, model):
    tokenized_events = []
    print("Tokenizing")
    for event in scan_events:
        print(event.subject_id, event.block, event.timestamp)
        # Tokenize
        sentences = tokenize_sentences(event.sentences, model)
        current_sentence = tokenize_text(event.current_sentence, model)
        stimulus = tokenize_text(event.stimulus, model)

        # Put tokenized version back (this could be done in one step, but it is more readable like this)
        event.sentences = sentences
        event.current_sentence = current_sentence
        event.stimulus = stimulus
        tokenized_events.append(event)

    return tokenized_events




def tokenize_sentences(sentences, model):
    tokenized_sents = []
    for sentence in sentences:
        tok_sent = model(sentence)
        tokenized_sents.append([tok.text for tok in tok_sent])
    return tokenized_sents


def tokenize_text(text, model):
    return [tok.text for tok in model(text)]
>>>>>>> upstream/master
