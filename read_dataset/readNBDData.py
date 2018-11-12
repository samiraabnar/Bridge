from os import listdir
import nibabel as nib
import numpy as np
from .scan import ScanEvent

# This method reads the Narrative Brain Dataset that was published by Lopopolo et al 2018.
# Paper: http://lrec-conf.org/workshops/lrec2018/W9/pdf/1_W9.pdf
# Data: https://osf.io/utpdy/

# It consists of fMRI data from 24 Dutch subjects who listened to three different narratives.
# Each subject went through 6 runs:
# The first three correspond to the narratives 1-3, run 4-6 correspond to reversed versions of the narratives.
# The presentation of the runs was randomized.
# TODO: WE should normalize the activation by the activation of the reverse runs.
# One scan was taken every 0.88 seconds.
# The stimuli have been aligned tokenwise with time stamps.

def read_all(data_dir):
    # --- READ TEXT STIMULI----
    stimuli = {}
    for narrative_id in range(1, 4):
        stimuli[narrative_id] = get_text_stimuli(data_dir, narrative_id)

    # --- PROCESS FMRI SCANS --- #
    all_events = []
    for subject in [file for file in listdir(data_dir + 'fmri/') if file.startswith('S')]:
        for narrative_id in range(1, 4):
            print("Processing data for subject: " + subject + " narrative: " + str(narrative_id))
            all_events.extend(read_run(data_dir, subject, narrative_id, stimuli[narrative_id]))

    return all_events


def read_run(data_dir, subject, narrative_id, stimuli):
    fmri_dir = data_dir + 'fmri/'

    # Run 1 corresponds to narrative 1
    current_dir = fmri_dir + subject + '/run' + str(narrative_id) + '/'

    # We use vswrs files, the letters correspond to preprocessing
    # “v” = motion correction, “s" = smoothing, “w" = normalisation, “r" = realign
    fmri_data = [file for file in listdir(current_dir) if
                 (file.endswith('.nii') and file.startswith('coreg_vswrs'))]

    # Each file corresponds to one scan taken every 0.88 seconds
    scan_time = 0.0
    word_index = 0
    word_time, word = stimuli[word_index]

    events = []
    sentences = []
    seen_text = ""

    for niifile in sorted(fmri_data):
        event = ScanEvent()
        # img.header can give a lot of metadata
        scan = nib.load(current_dir + niifile)
        image_array = np.asarray(scan.dataobj)

        # TODO This results in 510340 voxels, which seems to be too much! Analyze how this should be preprocessed
        voxel_vector = np.ndarray.flatten(image_array)

        # Get word sequence that has been played during previous and current scan
        word_sequence = ''
        while (word_time < scan_time) and (word_index + 1 < len(stimuli)):
            if len(word) > 0:
                if is_end_of_sentence(seen_text.strip()):
                    sentences.append(seen_text)
                    seen_text = word.strip() + " "

                # Simply add word to the current sentence
                else:
                    if len(word) > 0:
                        seen_text = seen_text + word.strip() + " "
                word_sequence += word + " "

            word_index += 1
            word_time, word = stimuli[word_index]
        word_sequence = word_sequence.replace('  ', ' ')

        # Set values of event
        event.stimulus = word_sequence.strip()
        event.sentences = list(sentences)
        event.current_sentence = seen_text
        event.subject_id = subject
        event.timestamp = scan_time
        event.scan = voxel_vector

        events.append(event)
        scan_time += 0.88

    return events


# --- PROCESS TEXT STIMULI --- #
# The text is already aligned with presentation times in a tab-separated file and ordered sequentially.
def get_text_stimuli(data_dir, narrative_id):
    text_dir = data_dir + 'text_metadata/'
    stimuli = []

    with open(text_dir + 'Narrative_' + str(narrative_id) + '_wordtiming.txt', 'r', encoding='utf-8',
              errors="replace") as textdata:
        for line in textdata:
            word, onset, offset, duration = line.strip().split('\t')
            stimuli.append([float(onset), word])
    return stimuli


# TODO: find out how to map voxel coordinates into regions

# Sentence boundary detection for the NBD data is very simple.
# There occur only . and ? at the end of a sentence and they never occur in the middle of a sentence.
def is_end_of_sentence(seentext):
    sentence_punctuation = (".", "?")
    if seentext.endswith(sentence_punctuation):
        return True
    else:
        return False
