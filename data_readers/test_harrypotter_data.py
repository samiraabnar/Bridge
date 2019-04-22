from data_readers.harrypotter_data import *
from language_preprocessing.tokenize import *

if __name__ == '__main__':

  # READ
  print("\n\nHarry Potter Data")
  harry_reader = HarryPotterReader(data_dir='/Users/samiraabnar/Codes/Data/harrypotter/')
  subject_id = 1
  harry_data = harry_reader.read_all_events(subject_ids=[subject_id])

  all_sentences = []
  tokenizer = SpacyTokenizer()
  tokenizer_fn = SpacyTokenizer.tokenize
  for block in harry_data[subject_id]:
    # These are all already sorted, so I think you don't even need timesteps.
    sentences = block.sentences
    scans = [event.scan for event in block.scan_events]
    stimuli = [event.stimulus_pointer for event in block.scan_events]
    timestamps = [event.timestamp for event in block.scan_events]

    print("\n\nBLOCK: " + str(block.block_id))
    print("Number of scans: " + str(len(scans)))
    print("Number of stimuli: " + str(len(stimuli)))
    print("Number of timestamps: " + str(len(timestamps)))
    print("Example stimuli 15-20 = Lists of (sentence_id, token_id): \n" + str(stimuli[14:20]))
    print("Example sentences 1-3: \n" + str(sentences[0:3]))

    context, index = block.get_stimuli_in_context(scan_event=block.scan_events[25],
                                                  tokenizer=tokenizer,
                                                  context_mode='sentence',
                                                  only_past=True)

    print("Stimuli Context:", context)
    print("Stimuli index:", index)
    print("new stimuli: ", np.asarray(context)[index])
    print("old stimuli: ", sentences[stimuli[25][0][0]][stimuli[25][0][1]])

    all_sentences.extend(block.get_processed_sentences(tokenizer=tokenizer))

  print(len(all_sentences))
  np.save("/Users/samiraabnar/Codes/Data/harrypotter/harrypotter_cleaned_sentences", all_sentences)
