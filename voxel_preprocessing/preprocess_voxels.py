import nilearn.signal
from scipy import stats

def clean(data):
    # check data format
    return nilearn.signal.clean(data, sessions=None,
                         detrend=True, standardize=True,
                         confounds=None, low_pass=None,
                         high_pass=0.005, t_r=2.0, ensure_finite=False)


# Double-check this
def zscore(data):
    return stats.zscore(data)

# TODO: normalize scans by resting state scans
