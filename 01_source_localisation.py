# import mne
import os.path as op
import argparse
import json
from utilities import files


# argparse input
des = "pipeline script"
parser = argparse.ArgumentParser(description=des)
parser.add_argument("-f", type=str, help="JSON file with pipeline parameters")
parser.add_argument("-n", type=int, help="id list index")
args = parser.parse_args()
params = vars(args)

# grab the cmd args
subj_index = params["n"]
json_file = params["f"]

# read the pipeline params
with open(json_file) as pipeline_file:
    pipeline_params = json.load(pipeline_file)

# establish paths
data_path = pipeline_params["data_path"]
mri_dir = op.join(data_path, "mri", "fs_results")
subjs = files.get_folders_files(mri_dir, wp=False)[0]
subjs.sort()
subj = subjs[:-1][subj_index]

mri_path = op.join(mri_dir, subj)
out_path = op.join(data_path, "stirring_source", subj)
files.make_folder(out_path)
raw_dir = op.join(data_path, "raw", subj)
trans_dir = op.join(data_path, "dig", subj)

# paths to files
raw_files = files.get_files(raw_dir,"", ".fif")[0]
raw_files.sort()
trans_file = files.get_files(trans_dir, subj, "trans.fif")[0][0]


if pipeline_params["coreg_viz"]:
    print("Alright, alright, alright.")