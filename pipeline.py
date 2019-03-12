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
import pandas as pd

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

mne.set_config("MNE_LOGGING_LEVEL", "CRITICAL")
verb = False

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
            verbose=verb
        )

        events = mne.find_events(
            raw, 
            stim_channel="UPPT001"
        )

        raw_resampled, events_resampled = raw.copy().resample(
            samp, 
            npad='auto', 
            events=events,
            n_jobs=-1,
        )

        mne.write_events(
            out_eve, 
            events_resampled
        )

        picks_meg = mne.pick_types(
            raw.info, 
            meg=True, 
            eeg=False, 
            eog=False, 
            ecg=False, 
            ref_meg=False
        )

        raw_resampled.filter(
            0,
            90,
            picks=picks_meg,
            n_jobs=-1,
            method="fir",
            phase="minimum"
        )

        tmin = (events_resampled[0,0] / samp) - 1
        tmax = (events_resampled[-1,0] / samp) + 1
        
        raw.crop(
            tmin=tmin,
            tmax=tmax
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

if pipeline_params["compute_ICA"]:
    exp_files = files.get_files(
        output_subj,
        "",
        "-raw.fif",
        wp=False
    )[0]
    exp_files.sort()

    for file in exp_files:
        raw_in = op.join(
            output_subj,
            file
        )

        ica_file = "{}-ica.fif".format(
            file[:-8],
        )

        ica_out = op.join(
            output_subj,
            ica_file
        )
        
        raw = mne.io.read_raw_fif(
            raw_in,
            preload=True,
            verbose=verb
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
        method = "extended-infomax"
        reject = dict(mag=4e-12)

        ica = ICA(
            n_components=n_components, 
            method=method
        )

        ica.fit(
            raw, 
            picks=picks_meg,
            reject=reject,
            verbose=verb
        )

        ica.save(ica_out)

if pipeline_params["create_JSON"]:
    """
    CREATE A JSON FILE TO FILL WHEN USING ICA_manual_inspection.py.
    RUN ONLY AFTER DOWNSAMPLING AND COMPUTING ICA. CLEARS THE FILE SO
    BE CAREFUL.
    """
    json_ICA = "ICA_comp.json"
    if not op.exists(json_ICA): # add not if the script is ready
        open(json_ICA, 'a').close()
    
    dict_JSON = {
        i: {} for i in subjs
    }

    for i in subjs:
        raws_in = op.join(
            output_path, 
            "data", 
            i
        )
        raw_files = files.get_files(
            raws_in,
            "",
            "-raw.fif",
            wp=False
        )[0]
        dict_JSON[i] = {x: [] for x in raw_files}
    files.dump_the_dict(
        json_ICA,
        dict_JSON
    )

if pipeline_params["apply_ICA"]:
    with open(json_ICA) as pipeline_file:
        ica_subj_comp = json.load(pipeline_file)

    for i in ica_subj_comp[subj].keys():
        
        rej_comps = ica_subj_comp[subj][i]
        raw_in = op.join(
            output_subj,
            i
        )
        ica_sol = op.join(
            output_subj,
            "{}-ica.fif".format(i[:-8])
        )

        cleaned_file = op.join(
            output_subj,
            "ica_cln_{}-raw.fif".format(i[:-8])
        )

        raw = mne.io.read_raw_fif(
            raw_in,
            preload=True,
            verbose=verb
        )

        ica = mne.preprocessing.read_ica(ica_sol)
        raw_ica = ica.apply(
            raw,
            exclude=rej_comps
        )
        
        raw_ica.save(
            cleaned_file,
            fmt="single",
            split_size="2GB",
            overwrite=True
        )
        
        print(cleaned_file, rej_comps)

if pipeline_params["filter_more"][0]:
    filter_iter = pipeline_params["filter_more"][1]
    raw_files = files.get_files(
        output_subj,
        "ica_cln",
        "-raw.fif",
        wp=False
    )[2]
    for hpass, lpass in filter_iter:
        folder_name = "filtered_{}_{}".format(hpass, lpass)
        folder_out = op.join(
            output_subj,
            folder_name
        )
        files.make_folder(folder_out)
        raw_files.sort()
        for i in raw_files:
            raw_in = op.join(
                output_subj,
                i
            )
            raw_out = op.join(
                output_subj,
                folder_name,
                i
            )

            raw = mne.io.read_raw_fif(
                raw_in,
                preload=True,
                verbose=verb
            )

            picks_meg = mne.pick_types(
                raw.info, 
                meg=True, 
                eeg=False, 
                eog=False, 
                ecg=False, 
                ref_meg=False
            )

            raw.filter(
                hpass,
                lpass,
                picks=picks_meg,
                n_jobs=-1,
                method="fir",
                phase="minimum"
            )

            raw.save(
                raw_out,
                fmt="single",
                split_size="2GB",
                overwrite=True
            )

            print(raw_out)

if pipeline_params["epochs"]:
    input_path = op.join(
        output_subj,
        pipeline_params["which_folder"]
    )

    beh_file = op.join(
        pipeline_params["beh_path"],
        subj,
        "resamp_beh.pkl"
    )

    exp_files = files.get_files(
        input_path,
        "ica_cln",
        "-raw.fif",
        wp=False
    )[2]

    event_files = files.get_files(
        output_subj,
        "",
        "-eve.fif",
        wp=False
    )[0]
    
    exp_files.sort()
    event_files.sort()

    event_files = [x for x in event_files if "rs" not in x]
    exp_files = [x for x in exp_files if "rs" not in x]

    events_d = {ef: mne.read_events(
        op.join(output_subj, ef),
        include=list(range(1,9))
    ) for ef in event_files}

    arr_lens = [events_d[i].shape[0] for i in event_files]
    arr_sect = np.cumsum(arr_lens)[:-1]
    all_evts = np.vstack(
        [events_d[i] for i in event_files]
    )
    all_evts = np.hstack(
        [all_evts, np.arange(all_evts.shape[0]).reshape(-1,1)]
    )
    new_evts = {f: e for f, e in zip(event_files, np.split(all_evts, arr_sect))}
    test_of_trueness = np.array(
        [sum(events_d[i][:,2] == new_evts[i][:,2]) for i in event_files]
    )
    test_of_length = arr_lens == test_of_trueness

    print(test_of_length)

    beh = pd.read_pickle(
        beh_file
    )

    beh_vs_meg = beh.conditions.values + 1 == all_evts[:,2]
    compatible = np.sum(beh_vs_meg) == all_evts.shape[0]

    print(compatible)

if pipeline_params["epoch_2s"]:

    for exp, eve in zip(exp_files, event_files):
        raw_in = op.join(
            input_path,
            exp
        )
        
        raw = mne.io.read_raw_fif(
            raw_in, 
            preload=True,
            verbose=verb
        )

        evts = events_d[eve]
        
        picks = mne.pick_types(
            raw.info, 
            meg=True, 
            eeg=False, 
            stim=False, 
            eog=False, 
            ref_meg='auto', 
            exclude='bads'
        ) 

        epochs1 = mne.Epochs(
            raw, 
            evts, 
            event_id=list(range(1, 9)),
            tmin=0.552, 
            tmax=2.752,
            picks=picks, 
            preload=True, 
            detrend=1, 
            baseline=(0.55,0.75),
            verbose=verb
            )
        
        epochs1.apply_baseline((0.552, 0.752))
        epochs1.shift_time(-0.752)
        epochs1.apply_baseline((-0.2, 0.0))

        epochs2 = mne.Epochs(
            raw,
            evts,
            event_id=list(range(1, 9)),
            tmin=2.552,
            tmax=4.752,
            picks=picks,
            preload=True,
            detrend=1,
            baseline=(2.552, 2.752),
            verbose=verb
        )

        epochs2.apply_baseline((2.552, 2.752))
        epochs2.shift_time(-2.752)
        epochs2.apply_baseline((-0.2, 0.0))

        epochs_file_out1 = op.join(
            input_path,
            "ph1_2s_{}-epo.fif".format(exp[8:-8])
        )

        epochs_file_out2 = op.join(
            input_path,
            "ph2_2s_{}-epo.fif".format(exp[8:-8])
        )

        epochs1.save(
            epochs_file_out1,
            split_size="2GB",
            fmt="single",
            verbose=False
        )
        print(epochs_file_out1)
        epochs2.save(
            epochs_file_out2,
            split_size="2GB",
            fmt="single",
            verbose=False
        )
        print(epochs_file_out2)

if pipeline_params["epoch_4s"]:
    for exp, eve in zip(exp_files, event_files):
        raw_in = op.join(
            input_path,
            exp
        )
        
        raw = mne.io.read_raw_fif(
            raw_in, 
            preload=True,
            verbose=verb
        )

        evts = events_d[eve]
        
        picks = mne.pick_types(
            raw.info, 
            meg=True, 
            eeg=False, 
            stim=False, 
            eog=False, 
            ref_meg='auto', 
            exclude='bads'
        ) 

        epochs = mne.Epochs(
            raw, 
            evts, 
            event_id=list(range(1, 9)),
            tmin=0.552, 
            tmax=4.752,
            picks=picks,
            baseline=(0.552,0.752),
            preload=True, 
            detrend=1,
            verbose=verb
            )
        epochs.apply_baseline((0.552, 0.752))
        epochs.shift_time(-0.752)
        epochs.apply_baseline((-0.2, 0.0))

        epochs_file_out = op.join(
            input_path,
            "4s_{}-epo.fif".format(exp[8:-8])
        )

        epochs.save(
            epochs_file_out,
            split_size="2GB",
            fmt="single",
            verbose=False
        )
        print(epochs_file_out)

if pipeline_params["epoch_response_locked"]:
    """
    """
    engage_ar = beh.engage_ix.values
    change_ar = beh.change_ix.values

    for exp, eve in zip(exp_files, event_files):
        # raw_in = op.join(
        #     input_path,
        #     exp
        # )
        
        # raw = mne.io.read_raw_fif(
        #     raw_in, 
        #     preload=True,
        #     verbose=verb
        # )

        evts = events_d[eve]
        engage_val = engage_ar[new_evts[eve][:,3]]
        
        evts[:,0] = evts[:,0] + engage_val

