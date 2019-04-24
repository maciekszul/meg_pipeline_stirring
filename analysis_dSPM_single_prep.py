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
import pandas as pd

"""
Single trial dSPM source localised, morphed to fsaverage. ICO5 respolution.
(c) 2019 Maciej Szul
"""


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
# garbage = ["001", "043", "042"]
# subjs = [i for i in subjs if i not in garbage]
subjs.sort()
subj = subjs[subj_index]

working_path = op.join(
    pipeline_params["output_path"],
    "data",
    subj,
    pipeline_params["which_processed_folder"])

subj_path = op.join(
    pipeline_params["output_path"],
    "data",
    subj
)

print(working_path)

noise_files = files.get_files(
    working_path,
    "mx",
    "-cov.fif",
    wp=True
)[2]
noise_files = [i for i in noise_files if "rs" not in i]
noise_files.sort()

fwd_files = files.get_files(
    subj_path,
    "",
    "-fwd.fif",
    wp=True
)[2]
fwd_files = [i for i in fwd_files if "rs" not in i]
fwd_files.sort()

epo_files = files.get_files(
    working_path,
    pipeline_params["which_processed_epochs"],
    "-epo.fif",
    wp=True
)[2]
epo_files.sort()

all_files = zip(noise_files, fwd_files, epo_files)

all_stc = []

for noise_f, fwd_f, epo_f in all_files:
    print(noise_f)
    print(fwd_f)
    print(epo_f)

    epo = mne.read_epochs(epo_f, preload=True)
    noise = mne.read_cov(noise_f)
    fwd = mne.read_forward_solution(fwd_f)
    inv = mne.minimum_norm.make_inverse_operator(
        epo.info,
        fwd,
        noise,
        depth=None,
        fixed="auto"
    )
    lambda2 = 1.0 / 3.0 ** 2
    stc = mne.minimum_norm.apply_inverse_epochs(
        epo,
        inv,
        lambda2,
        "dSPM",
        pick_ori=None
    )
    morph = mne.compute_source_morph(
        stc[0],
        spacing=4,
        subject_from=subj,
        subject_to="fsaverage",
        subjects_dir=fs_path)
    
    stc = [morph.apply(i) for i in stc]
    all_stc.extend(stc)

stc_file = op.join(
    working_path,
    "src_{}_{}.pickle".format(
        pipeline_params["which_processed_epochs"],
        "dSPM"
    )
)

with open(stc_file, "wb") as f:
    pickle.dump(all_stc, f)

named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
print("end:", time_string)