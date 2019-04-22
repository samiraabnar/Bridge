"""Util functions to deal with harry potter fMRI data."""



def is_beginning_of_new_sentence(sentence, newword):
  """This is a quite naive sentence boundary detection that only works for this dataset.
  """
  seentext = ' '.join(sentence)
  sentence_punctuation = (".", "?", "!", ".\"", "!\"", "?\"", "+")
  # I am ignoring the following exceptions, because they are unlikely to occur in fiction text:
  # "etc.", "e.g.", "cf.", "c.f.", "eg.", "al.
  exceptions = ("Mr.", "Mrs.")
  if seentext.endswith(exceptions):
      return False
  # This would not work if everything is lowercased!
  if seentext.endswith(sentence_punctuation) and not newword.islower() and newword is not ".":
      return True
  else:
      return False