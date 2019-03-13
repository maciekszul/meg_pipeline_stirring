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



subj = subjs[subj_index]

subj_path = op.join(
    data_out,
    subj
)

exp_files = files.get_files(
    subj_path,
    "ica_cln",
    "-raw.fif",
    wp=True
)[2]

exp_files.sort()
exp_file = exp_files[file_index]

trans_file = op.join(
    pipeline_params["data_path"],
    "dig",
    subj,
    "{}-trans.fif".format(subj)
)

raw = mne.io.read_raw_fif(
    exp_file,
    preload=False,
    verbose=False
)

mne.viz.plot_alignment(
    raw.info,
    trans_file, 
    subject=subj, 
    subjects_dir=pipeline_params["fs_path"], 
    surfaces="head-dense", 
    dig=True
)

mne.viz.plot_bem(
    subject=subj,
    subjects_dir=pipeline_params["fs_path"],
    brain_surfaces="white",
    orientation="coronal"
)