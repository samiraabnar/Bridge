import nilearn.signal
import numpy as np

def detrend(timeseries_datapoints, t_r, standardize=False):
    return nilearn.signal.clean(timeseries_datapoints, sessions=None,
                         detrend=True, standardize=standardize,
                         confounds=None, low_pass=None,
                         high_pass=0.005, t_r=t_r, ensure_finite=False)

def minus_average_resting_states(timeseries_datapoints, brain_states_with_no_stimuli):
  """
  :param timeseries_datapoints:
  :param brain_states_with_no_stimuli:
  :return:
  """

  # For now we simply normalize by minusing the avg resting state.
  average_brain_state_with_no_stimuli = np.mean(brain_states_with_no_stimuli, axis=-1)
  timeseries_datapoints = timeseries_datapoints - average_brain_state_with_no_stimuli

  return timeseries_datapoints


