import time
named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
print("start:", time_string)

import mne
import os.path as op
import argparse
import json
import pickle
import numpy as np
from tools import files
from scipy.stats import trim_mean, distributions
from operator import itemgetter
import pandas as pd
from functools import partial

"""
Preparation and permutation testing 
(c) 2019 Maciej Szul
"""

def where_mc(array, condition_list):
    """
    apply 1D np.where to list of multiple elements
    """
    out = np.hstack([np.where(array == cond) for cond in condition_list])[0]
    out.sort()
    return out


conditions = {
    0: 'hR_hR', 
    1: 'hR_hL', 
    2: 'hR_lR', 
    3: 'hR_lL', 
    4: 'hL_hR', 
    5: 'hL_hL', 
    6: 'hL_lR', 
    7: 'hL_lL'
}

bin_cond = {
    0: "no change of coherence and direction",
    1: "change of direction only",
    2: "change of coherence but no direction",
    3: "change of both coherence and direction"
}

json_file = "pipeline_params.json"

# argparse input
des = "pipeline script"
parser = argparse.ArgumentParser(description=des)
parser.add_argument(
    "-f", 
    type=str,
    default=json_file,
    help="JSON file with pipeline parameters"
)

parser.add_argument(
    "-n", 
    type=int, 
    help="id list index"
)

args = parser.parse_args()
params = vars(args)
json_file = str(params["f"])
subj_index = params["n"]

# read the pipeline params
with open(json_file) as pipeline_file:
    pipeline_params = json.load(pipeline_file)

# PATHS FOR WORK
fs_path = pipeline_params["fs_path"]
data_path = pipeline_params["data_path"]
output_path = pipeline_params["output_path"]

subjs = files.get_folders_files(pipeline_params["beh_path"], wp=False)[0]
garbage = ["001", "043", "042"]
subjs = [i for i in subjs if i not in garbage]
subjs.sort()

# list of values for the comparison of two conditions

dict_comp = {
        0: {"no_change": [0], "coh_change": [2]},
        1: {"no_change": [0], "dir_change": [1]},
        2: {"dir_change": [1], "dir_coh_change": [3]},
        3: {"coh_change": [2], "dir_coh_change": [3]},
}

comp = dict_comp[subj_index]

n_subjects = 13

X = {k: [] for k in comp.keys()}

for subj in subjs:
    stc_file = op.join(
        pipeline_params["output_path"],
        "data",
        subj,
        pipeline_params["which_processed_folder"],
        pipeline_params["which_data"]
    )

    beh_file = op.join(
        pipeline_params["beh_path"],
        subj,
        "resamp_beh.pkl"
    )

    beh = pd.read_pickle(beh_file)

    combined = beh.comb_cond.values


    with open(stc_file, "rb") as input_file:
        stc = pickle.load(input_file)

    X_s = {k: itemgetter(*where_mc(combined, comp[k]))(stc) for k in comp.keys()}
    X_s = {k: np.array([i.data for i in X_s[k]]) for k in X_s.keys()}
    X_s = {k: np.transpose(trim_mean(X_s[k], proportiontocut=0.2, axis=0)) for k in X_s.keys()}
    out = [X[k].append(X_s[k]) for k in X.keys()]

X = {k: np.array(X[k]) for k in X.keys()}

src = mne.setup_source_space(
    subject="fsaverage", 
    subjects_dir=fs_path, 
    spacing="ico4", 
    add_dist=False
)
connectivity = mne.spatial_src_connectivity(src)

tfce = dict(start=0, step=0.2)

clu = mne.stats.spatio_temporal_cluster_test(
    [X[k] for k in X.keys()],
    connectivity=connectivity,
    n_jobs=-1,
    threshold=tfce,
    verbose=True
)

output_path = op.join(
    pipeline_params["output_path"],
    "results",
    pipeline_params["which_processed_folder"]
)
files.make_folder(output_path)

name_ = list(comp.keys())

output_file = op.join(
    output_path,
    "{}_{}VS{}.pickle".format(pipeline_params["which_data"].split("_")[1], name_[0], name_[1])
)

with open(output_file, "wb") as f:
    pickle.dump(clu, f)

named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
print("end:", time_string)