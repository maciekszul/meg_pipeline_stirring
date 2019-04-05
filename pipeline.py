import time
named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
print("start:", time_string)

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
import copy

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
subjs.sort()
# subjs = subjs[:-1]
subj = subjs[subj_index]

raw_subj = op.join(data_path,"raw", subj)
dig_dir = op.join(data_path, "dig", subj)
output_subj = op.join(output_path, "data", subj)
files.make_folder(output_subj)

json_ICA = op.join(output_subj, "{}-ica-rej.json".format(subj))
samp = pipeline_params["downsample_raw"][1]

# mne.set_config("MNE_LOGGING_LEVEL", "CRITICAL")
verb = False

print("MNE-Python", mne.__version__)
print("subject: ", subj)
print("Freesurfer data path: ", fs_path)
print("subject output path: ", output_subj)



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

        print(ica_out)

if pipeline_params["apply_ICA"]:
    with open(json_ICA) as pipeline_file:
        ica_subj_comp = json.load(pipeline_file)

    for i in ica_subj_comp.keys():
        
        rej_comps = ica_subj_comp[i]
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
        folder_name = folder_name.translate(str.maketrans({".": ""}))
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

    print("test of length", test_of_length)

    beh = pd.read_pickle(
        beh_file
    )

    beh_vs_meg = beh.conditions.values + 1 == all_evts[:,2]
    compatible = np.sum(beh_vs_meg) == all_evts.shape[0]

    print("compatible", compatible)

if pipeline_params["make_epochs"]:
    """
    """
    engage_ar = beh.engage_ix.values
    change_ar = beh.change_ix.values

    # for exp, eve in zip([exp_files[0]], [event_files[0]]):
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

        meg_picks = mne.pick_types(
            raw.info, 
            meg=True, 
            eeg=False, 
            stim=False, 
            eog=False, 
            ref_meg="auto", 
            exclude='bads'
        )

        epochs_eng_path = op.join(
            input_path,
            "eng_{}-epo.fif".format(exp[8:-8])
        )

        epochs_ch_path = op.join(
            input_path,
            "ch_{}-epo.fif".format(exp[8:-8])
        )

        epochs_4s_path = op.join(
            input_path,
            "4s_{}-epo.fif".format(exp[8:-8])
        )

        epochs_ph1_path = op.join(
            input_path,
            "ph1_{}-epo.fif".format(exp[8:-8])
        )

        epochs_ph2_path = op.join(
            input_path,
            "ph2_{}-epo.fif".format(exp[8:-8])
        )

        evts_e = copy.copy(events_d[eve])
        engage_val = engage_ar[new_evts[eve][:,3]]
        evts_e[:,0] = evts_e[:,0] + engage_val

        evts_c = copy.copy(events_d[eve])
        change_val = change_ar[new_evts[eve][:,3]]
        evts_c[:,0] = evts_c[:,0] + change_val

        big_baseline = (0.552, 0.752)
        small_baseline = (-0.2, 0.0)
        big_epochs = mne.Epochs(
            raw,
            events_d[eve],
            event_id=list(range(1, 9)),
            tmin=0.552, 
            tmax=8.752,
            picks=meg_picks,
            baseline=((0.552, 0.752)),
            preload=True, 
            verbose=verb
        )
        big_epochs.apply_baseline(big_baseline)
        big_epochs.shift_time(-big_baseline[1])
        big_epochs.apply_baseline(small_baseline)

        epochs_4s = big_epochs.copy().crop(None, 4)
        epochs_4s.apply_baseline(None, None)
        epochs_4s.save(epochs_4s_path)
        del epochs_4s
        epochs_ph1 = big_epochs.copy().crop(None, 2)
        epochs_ph1.apply_baseline(None, None)
        epochs_ph1.save(epochs_ph1_path)
        del epochs_ph1
        epochs_ph2 = big_epochs.copy().crop(2, 4)
        epochs_ph2.apply_baseline(None, None)
        epochs_ph2.save(epochs_ph2_path)
        del epochs_ph2

        engage = []
        change = []
        for ix, evo in enumerate(list(big_epochs.iter_evoked())):
            en_evo = evo.copy()
            en_shift = engage_val[ix] / samp
            en_evo.crop(en_shift - 0.2, en_shift + 1.2)
            en_evo.shift_time(-en_shift)
            engage.append(en_evo)

            ch_evo = evo.copy()
            ch_shift = change_val[ix] / samp
            ch_evo.crop(ch_shift - 0.5, ch_shift + 1.5)
            ch_evo.shift_time(-ch_shift)
            change.append(ch_evo)

        engage_arr = np.array([i.data for i in engage])
        engage_epo = mne.EpochsArray(
            engage_arr,
            big_epochs.info,
            events_d[eve],
            tmin=-0.5,
        )
        engage_epo.apply_baseline(None, None)
        engage_epo.save(epochs_eng_path)
        del engage_epo

        change_arr = np.array([i.data for i in change])
        change_epo = mne.EpochsArray(
            change_arr,
            big_epochs.info,
            events_d[eve],
            tmin=-0.5,
        )
        change_epo.apply_baseline(None, None)
        change_epo.save(epochs_ch_path)
        del change_epo

        print("epochs_saved")

if pipeline_params["compute_forward_solution"]:

    src = mne.setup_source_space(
        subject=subj, 
        subjects_dir=fs_path, 
        spacing="oct6", 
        add_dist=False
    )

    src_file_out = op.join(
        output_subj,
        "{}-src.fif".format(subj)
    )

    mne.write_source_spaces(src_file_out, src, overwrite=True)

    conductivity = (0.3, )
    model = mne.make_bem_model(
        subject=subj,
        ico=5,
        conductivity=conductivity,
        subjects_dir=fs_path
    )

    bem = mne.make_bem_solution(model)

    bem_file_out = op.join(
        output_subj,
        "{}-bem.fif".format(subj)
    )

    mne.write_bem_solution(bem_file_out, bem)

    source_raw_files = files.get_files(
        output_subj,
        "ica_cln",
        "-raw.fif",
        wp=False
    )[2]
    source_raw_files.sort()

    trans_file = op.join(
        pipeline_params["data_path"],
        "dig",
        subj,
        "{}-trans.fif".format(subj)
    )
    
    for src_raw in source_raw_files:
        raw_in = op.join(
            output_subj,
            src_raw
        )

        fwd_out = op.join(
            output_subj,
            "{}-fwd.fif".format(src_raw[8:-8])
        )

        fwd = mne.make_forward_solution(
            raw_in,
            trans=trans_file,
            src=src,
            bem=bem,
            meg=True,
            eeg=False,
            mindist=5.0,
            n_jobs=-1
        )
        
        mne.write_forward_solution(
            fwd_out, 
            fwd, 
            verbose=verb, 
            overwrite=True
        )
        print(fwd_out)

if pipeline_params["compute_noise_covariance"]:
    input_path = op.join(
        output_subj,
        pipeline_params["which_folder"]
    )
    source_raw_files = files.get_files(
        input_path,
        "ica_cln",
        "-raw.fif",
        wp=False
    )[2]
    source_raw_files.sort()
    for src_raw in source_raw_files:
        raw_in = op.join(
            input_path,
            src_raw
        )

        cov_mx_out = op.join(
            input_path,
            "mx_{}-cov.fif".format(src_raw[8:-8])
        )

        raw = mne.io.read_raw_fif(
            raw_in, 
            preload=True,
            verbose=False
        )

        picks = mne.pick_types(
            raw.info, 
            meg=True, 
            eeg=False, 
            stim=False, 
            eog=False, 
            ref_meg="auto", 
            exclude="bads"
        )

        noise_cov = mne.compute_raw_covariance(
            raw, 
            method="auto", 
            rank=None,
            picks=picks,
            n_jobs=-1
        )

        noise_cov.save(
            cov_mx_out,
            overwrite=True
        )
        print(cov_mx_out)

if pipeline_params["compute_inverse_operator"]:
    input_path = op.join(
        output_subj,
        pipeline_params["which_folder"]
    )
    source_epo_files = files.get_files(
        input_path,
        pipeline_params["which_epochs"],
        "-epo.fif"
    )[2]
    source_epo_files.sort()
    fwd_solution_files = files.get_files(
        output_subj,
        "",
        "-fwd.fif"
    )[0]
    fwd_solution_files = [x for x in fwd_solution_files if "rs" not in x]
    fwd_solution_files.sort()
    
    cov_mx_files = files.get_files(
        input_path,
        "",
        "-cov.fif"
    )[0]
    cov_mx_files = [x for x in cov_mx_files if "rs" not in x]
    cov_mx_files.sort()

    for src, fwd, cov in zip(source_epo_files, fwd_solution_files, cov_mx_files):
        inv_oper_out = op.join(
            input_path,
            "{1}{0}-inv.fif".format(
                src[len(str(input_path))+len(pipeline_params["which_epochs"])+2:-8],
                pipeline_params["which_epochs"])
        )

        epochs = mne.read_epochs(
            src,
            preload=False,
            verbose=verb
        )

        fwd_data = mne.read_forward_solution(fwd)
        cov_data = mne.read_cov(cov)
        inverse_operator = mne.minimum_norm.make_inverse_operator(
            epochs.info,
            fwd_data,
            cov_data,
            loose=0.2,
            depth=0.8
        )

        mne.minimum_norm.write_inverse_operator(
            inv_oper_out,
            inverse_operator
        )
        print(inv_oper_out)

if pipeline_params["compute_inverse_solution"][0]:
    method_dict = {
        "dSPM": (8, 12, 15),
        "sLORETA": (3, 5, 7),
        "eLORETA": (0.75, 1.25, 1.75)
    }

    input_path = op.join(
        output_subj,
        pipeline_params["which_folder"]
    )
    epo_files = files.get_files(
        input_path,
        pipeline_params["which_epochs"],
        "-epo.fif"
    )[2]
    epo_files.sort()
    inv_files = files.get_files(
        input_path,
        pipeline_params["which_epochs"],
        "-inv.fif"
    )[2]
    inv_files.sort()
    print(epo_files)
    print(inv_files)
    method = pipeline_params["compute_inverse_solution"][1]
    snr = 3.
    lambda2 = 1. / snr ** 2
    lims = method_dict[method]

    for epo_path, inv_path in zip(epo_files, inv_files):
        epo = mne.read_epochs(
            epo_files,
            verbose=verb,
            preload=True
        )
        inv = mne.minimum_norm.read_inverse_operator(
            inv_path,
            verbose=verb
        )
        
        stc = mne.minimum_norm.apply_inverse_epochs(
            epochs,
            inverse_operator,
            lambda2,
            method=method,
            pick_ori=None,
            verbose=True
        )

named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
print("end:", time_string)
