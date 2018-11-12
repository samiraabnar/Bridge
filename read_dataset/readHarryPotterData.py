import numpy as np
import scipy.io
from read_dataset.scan import ScanEvent
from read_dataset.brain_data_reader import BrainDataReader
# This method reads the Harry Potter data that was published by Wehbe et al. 2014
# Paper: http://aclweb.org/anthology/D/D14/D14-1030.pdf
# Data: http://www.cs.cmu.edu/afs/cs/project/theo-73/www/plosone/


# It consists of fMRI data from 8 subjects who read chapter 9 of the first book of Harry Potter.
# They see one word every 0.5 seconds.
# A scan is taken every two seconds.
# The chapter was presented in four blocks of app. 12 minutes.
# Voxel size: 3 x 3 x 3

class HarryPotterReader(BrainDataReader):

  def __init__(self, data_dir):
    super(HarryPotterReader, self).__init__(data_dir)

  def read_all_events(self, subject_ids=None):
    # Collect scan events
    events = {}
    if subject_ids is None:
      subject_ids = np.arange(1, 9)
    for subject_id in subject_ids:
      events[subject_id] = {}
      for block_id in np.arange(1, 5):
        print("reading subject %d, block %d" % (subject_id, block_id))
        events[subject_id][block_id] = self.read_block(subject_id, block_id, voxel_mapping=True)

    return events

  def read_block(self, subject_id, block_id, voxel_mapping=True):

    # Load matlab file for the given subject:
    datafile = scipy.io.loadmat(self.data_dir + "subject_" + str(subject_id) + ".mat")

    # Data structure is a dictionary with keys data, time, words, meta
    # Shapes for subject 1, block 1: data (1351,37913), time (1351,2) words (1, 5176)
    noisy_scans = datafile["data"]
    timedata = datafile["time"]
    presented_words = datafile["words"]

    # --- PROCESS FMRI SCANS --- #
    # We have one scan every 2 seconds
    scan_times = timedata[:, 0]

    # We have four blocks. One block includes approx. 12 minutes
    blocks = timedata[:, 1]

    # find first and last scan time of current block
    block_start = np.min(scan_times[np.where(blocks == block_id)])
    block_end = np.max(scan_times[np.where(blocks == block_id)])


    # --- PROCESS TEXT STIMULI -- #
    time_word_tuples = self.extract_block_stimuli(presented_words, block_start, block_end)

    # Initialize variables
    # stimulus = words presented between current and previous scan
    # noisy_scan = voxel activations (with the standard preprocessing: motion correction, slice timing correction etc, but not yet cleaned).
    # Details about the preprocessing can be found in the Appendix of Wehbe et al.
    # We save everything in arrays because the data is already ordered.
    word_index = 0

    word_time = time_word_tuples[word_index][0]
    word = time_word_tuples[word_index][1]

    events = []
    sentences = []
    seen_text = ""
    start = np.where(scan_times == block_start)[0][0]
    end = np.where(scan_times == block_end)[0][0]
    voxel_mapping = self.get_voxel_to_region_mapping(datafile['meta'])
    for j in np.arange(start, end+1):
      event = ScanEvent()
      scan_time = scan_times[j]

      word_sequence = ""

      # Collect the words that have been represented during the previous and the current scan.
      while (word_time < scan_time) and ((word_index + 1) < len(time_word_tuples)):
        # Words are currently stored exactly as presented, preprocessing can be done later
        if len(word) > 0:
          seen_text = seen_text + word.strip() + " "
        word_sequence += word + " "

        # Get next word
        word_index += 1
        word_time, word = time_word_tuples[word_index]

      # reached next scan, add collected data to events
      event.subject_id = str(subject_id)
      event.block = block_id
      event.timestamp = scan_time
      event.scan = noisy_scans[j]

      event.stimulus = word_sequence.strip()
      event.sentences = list(sentences)
      event.current_sentence = seen_text
      event.voxel_to_region_mapping = voxel_mapping

      events.append(event)

    # Add the last sentence to the sentences of the last event.
    events[-1].sentences.append(seen_text.strip())
    return events

  def extract_block_stimuli(self, presented_words, block_start, block_end):
    """ Here we extract the presented words and align them with their timestamps.
        The original data consists of weirdly nested arrays.

    """
    time_word_tuples = []

    for i in np.arange(presented_words.shape[1]):
      token = presented_words[0][i][0][0][0][0]
      timestamp = presented_words[0][i][1][0][0]
      if timestamp < block_end and timestamp >= block_start:
        time_word_tuples.append([timestamp, token])

    return time_word_tuples

# The metadata provides very rich information.
# Double-check description.txt in the original data.
# Important: Each voxel in the scan has different coordinates depending on the subject!
# Voxel 5 has the same coordinates in all scans for subject 1.
# Voxel 5 has the same coordinates in all scans for subject 2, but they differ from the coordinates for subject 1.
# Same with regions: Each region spans a different set of voxels depending on the subject!
  def get_voxel_to_region_mapping(self, metadata):
    roi_of_nth_voxel = metadata[0][0][8][0]
    roi_names = metadata[0][0][10][0]
    voxel_to_region = {}
    for voxel in np.arange(0, roi_of_nth_voxel.shape[0]):
      roi = roi_of_nth_voxel[voxel]
      voxel_to_region[voxel] = roi_names[roi][0]
    # for name in roi_names:
    #  print(name[0])
    return voxel_to_region

# --------
# These are some lines for processing the metadata which are not needed here, but I leave them in for reference.
# Setting indices according to description.txt in the original data folder
# sub_id_index = 0
# number_of_scans_index = 1
# number_of_voxels = 2
# x_dim_index = 3
# y_dim_index = 4
# z_dim_index = 5
# colToCoord_index = 6
# coordToCol_index = 7
# ROInumToName_index = 8
# ROInumsToName_3d_index = 9
# ROINames_index = 10
# voxel_size_index = 11
# matrix_index = 12

# Extract metadata
# Number of scans is constant over all subjects: 1351
# Voxel size is also constant 3x3x3
# Number of voxels varies across subjects.
# the_subject_id = metadata[0][0][sub_id_index][0][0]
# number_of_scans = metadata[0][0][number_of_scans_index][0][0]
# number_of_voxels = metadata[0][0][number_of_voxels][0][0]
# voxel_size = metadata[0][0][voxel_size_index]

# Example: get coordinates of 5th voxel for this subject
# coordinates_of_nth_voxel = metadata[0][0][6]
# coordinates_of_nth_voxel[5]
# which_voxel_for_coordinates = metadata[0][0][7]
# get voxel number for set of coordinates
# which_voxel_for_coordinates[36,7,19]

#  These coordinates differ slightly across subjects
# x_dim = metadata[0][0][x_dim_index][0][0]
# y_dim = metadata[0][0][y_dim_index][0][0]
# z_dim = metadata[0][0][z_dim_index][0][0]

# This index provides the geometric coordinates (x,y,z) of the n-th voxel
# coords = metadata[0][0][colToCoord_index]

# I am not really sure how voxels are mapped into Regions of Interest
# column_map = metadata[0][0][coordToCol_index]
# nmi_matrix = metadata[0][0][matrix_index]
# named_areas = metadata[0][0][ROInumToName_index]
# area_names_list = metadata[0][0][ROINames_index][0]

# The Appendix.pdf gives information about the fMRI preprocessing (slice timing correction etc)
# Poldrack et al 2014: The goal of spatial normalization is to transform the brain images from each individual in order to reduce the variability between individuals and allow
# meaningful group analyses to be successfully performed.

# TODO ask Samira: Wehbe et al used a Gaussian kernel smoother and voxel selection, you too?

# Signal Drift: : a global signal decrease with subsequently acquired images in the scan (technological artifact)
# Detrending? Tanabe & Meyer 2002:
# Because of the inherently low signal to noise ratio (SNR) of fMRI data, removal of low frequency signal
# intensity drift is an important preprocessing step, particularly in those brain regions that weakly activate.
# Two known sources of drift are noise from the MR scanner and aliasing of physiological pulsations. However,
# the amount and direction of drift is difficult to predict, even between neighboring voxels.

# from nilearn documentation:
# Standardize signal: set to unit variance
# low_pass, high_pass: Respectively low and high cutoff frequencies, in Hertz.
# Low-pass filtering improves specificity.
# High-pass filtering should be kept small, to keep some sensitivity.


if __name__ == '__main__':
  brain_data_reader = HarryPotterReader(data_dir='/Users/samiraabnar/Codes/Data/harrypotter/')
  all_events = brain_data_reader.read_all_events(subject_ids=[1])

  print(all_events[1][1][-1].__dict__)