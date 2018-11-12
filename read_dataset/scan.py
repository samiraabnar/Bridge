# The ScanEvent class should be a general object for all kinds of brain data
# Currently, we only use fMRI, but it could be extended.
# The required fields are a current guess and are likely to be modified later on.
class ScanEvent:

    subject_id = ""

    # Experimental block or run
    block = 1

    # All tokens that have been presented between the previous and the current scan
    stimulus = []

    # All complete sentences that have been presented during the block up to the previous scan:
    sentences = []

    # The sentence we are currently in:
    current_sentence = ""

    timestamp = 0.0
    scan = []

    # If available: a mapping from the nth-voxel to the corresponding brain region
    voxel_to_region_mapping = {}

