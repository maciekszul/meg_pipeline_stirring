import mne
from mne.preprocessing import ICA
import os.path as op
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from tools import files
import subprocess as sp

json_file = "pipeline_params.json"

# argparse input
des = "pipeline script"
parser = argparse.ArgumentParser(description=des)
parser.add_argument(
    "-f", 
    type=str, 
    nargs=1,
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
json_file = params["f"]
subj_index = params["n"]

# read the pipeline params
with open(json_file) as pipeline_file:
    pipeline_params = json.load(pipeline_file)

# PATHS FOR WORK
fs_path = pipeline_params["fs_path"]
data_path = pipeline_params["data_path"]
output_path = pipeline_params["output_path"]

subjs = files.get_folders_files(fs_path, wp=False)[0]
subjs.sort()
subjs = subjs[:-1]
subj = subjs[subj_index]

raw_subj = op.join(data_path,"raw", subj)
dig_dir = op.join(data_path, "dig", subj)
output_subj = op.join(output_path, "data", subj)
files.make_folder(output_subj)

json_ICA = "ICA_comp.json"


if pipeline_params["downsample_raw"][0]:
    """
    DOWNSAMPLE THE RAW *.FIF FILES
    AND EVENTS. SAVE IN THE OUTPUT DIR
    """
    exp_files = files.get_files(
        raw_subj,
        "",
        "-raw.fif",
        wp=False
    )[0]
    exp_files.sort()
    samp = pipeline_params["downsample_raw"][1]

    for filename in exp_files:
        raw_path = op.join(
            raw_subj,
            filename
        )
        
        out_raw = op.join(
            output_subj,
            filename
        )

        eve_name = "{}-eve.fif".format(
            filename[:-8]
        )

        out_eve = op.join(
            output_subj,
            eve_name
        )

        raw = mne.io.read_raw_fif(
            raw_path, 
            preload=True, 
            verbose=False
        )

        events = mne.find_events(
            raw, 
            stim_channel="UPPT001"
        )

        raw_resampled, events_resampled = raw.copy().resample(
            samp, 
            npad='auto', 
            events=events,
            n_jobs=-1
        )

        mne.write_events(
            out_eve, 
            events_resampled
        )

        raw_resampled.save(
            out_raw, 
            fmt="single", 
            split_size="2GB", 
            overwrite=True
        )

        print(raw_path)
        print(out_raw)
        print(out_eve)

if pipeline_params["filter_raw"][0]:
    """
    FILTER AND DETREND RAW FILES USING FILTERS SPECIFIED IN PARAMETERS
    """
    import itertools as it

    exp_files = files.get_files(
        output_subj,
        "",
        "-raw.fif",
        wp=False
    )[0]
    exp_files.sort()

    filter_iter = pipeline_params["filter_raw"][1]
    hp_lp_iter = it.product(filter_iter, exp_files)

    for (hpass, lpass), file in hp_lp_iter:
        raw_in = op.join(
            output_subj,
            file
        )

        filter_file = "{}_{}_{}".format(
            hpass,
            lpass,
            file
        )

        filter_out = op.join(
            output_subj,
            filter_file
        )

        raw = mne.io.read_raw_fif(
            raw_in,
            preload=True,
            verbose=False
        )

        picks_meg = mne.pick_types(
            raw.info, 
            meg=True, 
            eeg=False, 
            eog=False, 
            ecg=False, 
            ref_meg=False
        )
        
        raw = raw.filter(
            hpass,
            lpass,
            picks=picks_meg,
            n_jobs=-1,
            method="fir",
            phase="minimum"
        )

        raw.save(filter_out, fmt="single", split_size="2GB")
        print(hpass, lpass, filter_file)


if pipeline_params["create_ICA_json"]:
    """
    TO DO: json generation should be subject specific (amount of files)
    """
    filter_iter = pipeline_params["filter_raw"][1]
    list_of_files = []
    for i in filter_iter:
        prefix = "{}_{}".format(*i)
        print(prefix)
        exp_files = files.get_files(
            output_subj,
            prefix,
            "-raw.fif",
            wp=False
        )[1]
        exp_files.sort()
        list_of_files.extend(exp_files)
    
    if not os.path.exists(json_ICA): # add not if the script is ready
        open(json_ICA, 'a').close()
        ica_json = {i:None for i in subjs}
        for key in ica_json.keys():

            ica_json[key] = {i: [] for i in list_of_files}

        files.dump_the_dict(json_ICA, ica_json)

if pipeline_params["compute_ICA"][0]:
    with open(json_ICA) as pipeline_file:
        for_ICA = json.load(pipeline_file)
    
    filter_iter = pipeline_params["compute_ICA"][1]
    files_for_ICA = list(for_ICA[subj].keys())

    to_compute = []
    for i in filter_iter:
        prefix = "{}_{}".format(*i)
        files_freq = files.items_cont_str(files_for_ICA, prefix)
        to_compute.extend(files_freq)

    for file in to_compute:
        raw_in = op.join(
            output_subj,
            file
        )
        ica_out = op.join(
            output_subj,
            "{}-ica.fif".format(file[:-8])
        )
        print(ica_out)
        raw = mne.io.read_raw_fif(
            raw_in,
            preload=True,
            verbose=False
        )

        picks_meg = mne.pick_types(
            raw.info, 
            meg=True, 
            eeg=False, 
            eog=False, 
            ecg=False, 
            ref_meg=False
        )

        n_components = 50
        method = "fastica"
        decim = None
        random_state = None
        reject = dict(mag=5e-12)

        ica = ICA(
            n_components=n_components, 
            method=method, 
            random_state=random_state
        )

        ica.fit(
            raw, 
            picks=picks_meg, 
            decim=decim, 
            reject=reject
        )
        ica.save(ica_out)