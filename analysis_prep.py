import time
named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
print("start:", time_string)

import mne
import os.path as op
import argparse
import json
import numpy as np
from tools import files
import pandas as pd

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



named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
print("end:", time_string)