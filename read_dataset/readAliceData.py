from os import listdir
from .scan import ScanEvent

# This method reads the Alice data that was published by John Brennan et al. 2016
# Paper: https://www.ncbi.nlm.nih.gov/pubmed/27208858
# Data: https://drive.google.com/file/d/0By_8Ci8eoDI4Q3NwUEFPRExIeG8/view
# Stimuli: https://drive.google.com/file/d/0By_8Ci8eoDI4d2I4bEF1VEZKZjA/view

# It consists of fMRI data from 28 subjects who listen to chapter 1 of Alice in Wonderland.
# We skip subject 33 because the data is corrupted.
# The raw fMRI data is not yet available, we only have averages for 6 regions of interests (ROIs).
# The ROI size can be 6 or 10.

def read_all(data_dir, roi_size):
    all_events = []

    # Read in the data
    for alice_subject in listdir(data_dir + 'alice_data_shared/' + str(roi_size) + 'mm/'):
        if alice_subject == 's33-timecourses.txt':
            continue
        all_events.extend(read(data_dir, alice_subject, roi_size))

    # Add voxel to region mapping, it is the same for every participant here.
    mapping = get_voxel_to_region_mapping()
    for event in all_events:
        event.voxel_to_region_mapping = mapping

    # All_events is a list of all scan events in the data. They can be sorted by subject_id, block, timestamp etc
    return all_events

def read(data_dir, subject, roi_size):
    xmin = 0.0
    fmri_dir = data_dir + 'alice_data_shared/' + str(roi_size) + 'mm/'
    textdata_file = data_dir + 'alice_stim_shared/DownTheRabbitHoleFinal_exp120_pad_1.TextGrid'

    # --- PROCESS TEXT STIMULI --- #
    # This is a textgrid file. Processing is not very elegant, but works for this particular example.
    # The text does not contain any punctuation except for apostrophes, only words!
    # Apostrophes are separated from the previous word, maybe I should remove the whitespace for preprocessing?
    # We have 12 min of audio/text data.
    # We save the stimuli in an array because word times are already ordered
    stimuli = []

    with open(textdata_file, 'r') as textdata:
        i = 0
        for line in textdata:
            if i > 13:
                line = line.strip()
                # xmin should always be read BEFORE word
                if line.startswith("xmin"):
                    xmin = float(line.split(" = ")[1])
                if line.startswith("text"):
                    word = line.split(" = ")[1].strip("\"")
                    # Praat words: "sp" = speech pause, we use an empty stimulus instead
                    if word == "sp":
                        word = ""
                    stimuli.append([xmin, word.strip()])
            i += 1

        # --- PROCESS FMRI SCANS --- #
        # We have 361 fmri scans.
        # They have been taken every two seconds.
        # One scan consists of entries for 6 regions --> much more condensed data than Harry Potter


    with (open(fmri_dir + subject, 'r')) as subjectdata:
        scans = []
        for line in subjectdata:
            # Read activation values from file
            activation_strings = line.strip().split("   ")

            # Convert string values to floats
            activations = []
            for a in activation_strings:
                activations.append(float(a))

            scans.append(activations)

    # --- ALIGN STIMULI AND SCANS AS EVENTS
    # stimulus = words presented between current and previous scan
    # scan = accumulated voxel activations for 6 regions

    scan_time = 0.0
    word_index = 0
    word_time, word = stimuli[word_index]
    events = []

    for scan in scans:
        event = ScanEvent()

        # Collect all words presented during current scan
        stimulus = ''
        while word_time < scan_time:
            stimulus += word.strip() + ' '
            stimulus = stimulus.replace('  ', ' ')
            word_index += 1
            word_time, word = stimuli[word_index]

        # TODO: Detect sentence boundaries and collect sentences seen so far.
        # TODO: There is no punctuation in the stimulus data, we need to get it from here: https://www.cs.cmu.edu/~rgs/alice-I.html

        # Set information for current scan event
        event.stimulus = stimulus.strip()
        event.subject_id = subject.split('-')[0]
        event.timestamp = scan_time
        event.scan = scan

        events.append(event)
        scan_time += 2

    return events


# Note that region names are not the same as for the Wehbe data!
# The abbreviations stand for:
# LATL: left anterior temporal lobe
# RATL: right anterior temporal lobe
# LPTL: left posterior temporal lobe
# LIPL: left inferior parietal lobe
# LPreM: left premotor
# LIFG: left inferior frontal gyrus

def get_voxel_to_region_mapping():
    return {0: "LATL", 1: "RATL",  2: "LPTL", 3: "LIPL", 4: "LPreM", 5: "LIFG",}
