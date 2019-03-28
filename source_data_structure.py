import mne
import os.path as op
import numpy as np
import json
from tools import files
import pandas as pd

fs = "/cubric/data/c1557187/stirring_MEG/mri/fs_results"
subj = "021"

path = "/cubric/scratch/c1557187/stirring_source/data/021/filtered_01_40/"

verb = True

inv = op.join(path, "ch_1-inv.fif")
epo = op.join(path, "4s_1-epo.fif")

epochs = mne.read_epochs(
    epo,
    preload=False,
    verbose=verb
)

inverse = mne.minimum_norm.read_inverse_operator(
    inv,
    verbose=verb
)

method_dict = {
    "dSPM": (8, 12, 15),
    "sLORETA": (3, 5, 7),
    "eLORETA": (0.75, 1.25, 1.75)
}

method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2
lims = method_dict[method]

evo = epochs.average()
evo

stc = mne.minimum_norm.apply_inverse(
    evo,
    inverse,
    lambda2,
    method=method,
    pick_ori=None,
    verbose=True
)

# stc.plot(
#     subjects_dir=fs,
#     subject=subj,
#     hemi="both",
#     time_viewer=True
# )