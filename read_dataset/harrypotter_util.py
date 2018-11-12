"""Util functions to deal with harry potter fMRI data."""

def is_beginning_of_new_sentence(seentext, newword):
  """This is a quite naive sentence boundary detection that only works for this dataset.

  :param seentext:
  :param newword:
  :return:
  """
  sentence_punctuation = (".", "?", "!", ".\"", "!\"", "?\"", "+", "…\"")
  # I am ignoring the following exceptions, because they are unlikely to occur in fiction text: "etc.", "e.g.", "cf.", "c.f.", "eg.", "al.
  exceptions = ("Mr.", "Mrs. ")
  # This would not work if everything is lowercased!
  if (seentext.endswith(sentence_punctuation)) and not seentext.endswith(exceptions) and not newword.islower():
    return True
  else:
    return False
