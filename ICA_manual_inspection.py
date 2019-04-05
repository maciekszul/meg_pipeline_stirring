import mne
import os.path as op
import argparse
import json
import numpy as np
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
from tools import files


# argparse input
des = "pipeline script"
parser = argparse.ArgumentParser(description=des)
parser.add_argument(
    "-n", 
    type=int, 
    help="id list index"
)

parser.add_argument(
    "-f", 
    type=int, 
    help="file list index"
)

args = parser.parse_args()
params = vars(args)
subj_index = params["n"]
file_index = params["f"]

json_ICA = "ICA_comp.json"

json_file = "pipeline_params.json"
# read the pipeline params
with open(json_file) as pipeline_file:
    pipeline_params = json.load(pipeline_file)


# PATHS
data_out = op.join(
    pipeline_params["output_path"],
    "data"
)

subjs = files.get_folders_files(
    data_out,
    wp=False
)[0]

subjs.sort()

subj = subjs[subj_index]

subj_path = op.join(
    data_out,
    subj
)

ica_files = files.get_files(
    subj_path,
    "",
    "-ica.fif"
)[2]
ica_files.sort()
raw_files = files.get_files(
    subj_path,
    "",
    "-raw.fif",
    wp=True
)[2]
raw_files.sort()

raw_file = raw_files[file_index]
ica_file = ica_files[file_index]

comp_ICA_json_path = op.join(
    subj_path,
    "{}-ica-rej.json".format(str(subj).zfill(3))
)

if not op.exists(comp_ICA_json_path):
    json_dict = {
        op.split(i)[1]: [] for i in raw_files
    }
    files.dump_the_dict(comp_ICA_json_path, json_dict)


raw = mne.io.read_raw_fif(raw_file , preload=True, verbose=False)

set_ch = {'EEG057-3305':'eog', 'EEG058-3305': 'eog'}
raw.set_channel_types(set_ch)

ica = mne.preprocessing.read_ica(ica_file)

eog_ix, eog_scores = ica.find_bads_eog(
    raw, 
    threshold=3.0, 
    l_freq=1, 
    h_freq=10, 
    verbose=False
)
eog_ix.sort()
print(subj)
print(eog_ix)

ica.plot_scores(eog_scores, exclude=eog_ix)

ica.plot_sources(raw)
