import time
named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
print("start:", time_string)
import mne
from mne.preprocessing import ICA
import os.path as op
import json
from tools import files
import numpy as np
import pandas as pd
import sys


# parsing command line arguments
try:
    index = int(sys.argv[1])
except:
    print("incorrect arguments")
    sys.exit()

try:
    json_file = sys.argv[2]
    print(json_file)
except:
    json_file = "pipeline.json"
    print(json_file)

# open json file
with open(json_file) as pipeline_file:
    parameters = json.load(pipeline_file)

# filters
filter_list = {
        "time": (0.1, 40),
        "delta": (0.1, 4),
        "theta": (4, 8),
        "alpha": (8, 14),
        "beta": (14, 30),
        "gamma": (30, 90)
    }

# prepare paths
raw_path = parameters["raw_path"]
out_path = parameters["path"]

# subjects
subjects = files.get_folders_files(
    raw_path,
    wp=False
)[0]

exclude = ["036", "043"]

subjects = [i for i in subjects if i not in exclude]
subjects.sort()

subject = subjects[index]

raw_subject_dir = op.join(
    raw_path,
    subject
)
meg_subject_dir = op.join(
    out_path,
    "MEG",
    subject
)

files.make_folder(meg_subject_dir)

beh_subject_path = op.join(
    out_path,
    "BEH",
    "beh_{}_matched.gz".format(subject)
)

if parameters["step_1"]:
    raw_files = files.get_files(
        raw_subject_dir,
        "",
        "-raw.fif"
    )[2]

    raw_files = [i for i in raw_files if "_rs" not in i]
    raw_files.sort()

    for ix, raw_path in enumerate(raw_files):
        file_ix = str(ix).zfill(3)
        print(raw_path)
        raw = mne.io.read_raw_fif(
            raw_path,
            preload=True
        )
        set_ch = {"EEG057-3305":"eog", "EEG058-3305": "eog", "UPPT001": "stim"}
        raw.set_channel_types(set_ch)

        raw = raw.pick_types(
            meg=True,
            ref_meg=True,
            eog=True,
            eeg=False,
            stim=True
        )

        events = mne.find_events(
            raw,
            min_duration=0.003
        )

        crop_min = events[0][0]/ raw.info['sfreq'] - 2
        crop_max = events[-1][0]/ raw.info['sfreq'] + 2
        raw.crop(tmin=crop_min, tmax=crop_max)

        filter_picks = mne.pick_types(
            raw.info,
            meg=True,
            ref_meg=True,
            stim=False,
            eog=False
        )

        low_freq, high_freq  = (0.1, 90)
        raw_output_path = op.join(
            meg_subject_dir,
            "time-frequency-{}-raw.fif".format(file_ix),
        )
        events_output_path = op.join(
            meg_subject_dir,
            "eve-{}-eve.fif".format(file_ix)
        )

        ica_output_path = op.join(
            meg_subject_dir,
            "{}-ica.fif".format(file_ix)
        )

        raw = raw.filter(
            low_freq,
            None,
            method="fir",
            phase="minimum",
            n_jobs=-1,
            picks=filter_picks
        )
        
        raw = raw.filter(
            None,
            high_freq,
            method="fir",
            phase="minimum",
            n_jobs=-1,
            picks=filter_picks
        )

        raw, events = raw.copy().resample(
            250, 
            npad="auto", 
            events=events,
            n_jobs=-1,
        )

        # ANNOTATIONS TO EXCLUDE JOYSTICK PARTS FROM ICA FITTING
        onsets_p2 = mne.pick_events(events, include=list(np.arange(10,18)))
        annot_onset = ((onsets_p2[:,0] - raw.first_samp) / raw.info["sfreq"]) - 2
        duration = np.array([4.0] * annot_onset.shape[0])
        description = np.array(["bad_joystick_movement"] * annot_onset.shape[0])

        annotations = mne.Annotations(
            annot_onset,
            duration,
            description
        )
        raw.set_annotations(annotations)
        
        # ICA
        n_components = 50
        method = "fastica"
        max_iter = 10000

        ica = ICA(
            n_components=n_components, 
            method=method,
            max_iter=max_iter
        )

        ica.fit(
            raw,
            reject_by_annotation=True
        )

        ica.save(ica_output_path)
        print("ICA saved")

        raw.annotations.delete(list(range(80)))

        raw.save(raw_output_path, overwrite=True)
        print("RAW saved")
        mne.write_events(events_output_path, events)
        print("Events saved")
        
        named_tuple = time.localtime() # get struct_time
        time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
        print("step 1 done:", time_string)


if parameters["step_2"]:
    raw_files = files.get_files(
        meg_subject_dir,
        "time-frequency",
        "-raw.fif"
    )[2]
    raw_files.sort()

    ica_files = files.get_files(
        meg_subject_dir,
        "",
        "-ica.fif"
    )[2]
    ica_files.sort()

    eve_files = files.get_files(
        meg_subject_dir,
        "",
        "-eve.fif"
    )[2]
    eve_files.sort()

    beh = pd.read_pickle(
        beh_subject_path
    )

    beh = beh.sort_values(["run", "trial"])
    runs = beh.run.unique()
    runs.sort()

    components_file_path = op.join(
        meg_subject_dir,
        "rejected-components.json"
    )

    with open(components_file_path) as data:
        components_rej = json.load(data)
    
    iter_ = zip(raw_files, ica_files, eve_files)
    
    # containers for the epochs
    long_epochs = []
    for raw_path, ica_path, eve_path in iter_:
        key = raw_path.split("/")[-1]
        file_ix = key.split("-")[2]
        print(key)
        print(file_ix)

        raw = mne.io.read_raw_fif(
            raw_path,
            preload=True
        )

        ica = mne.preprocessing.read_ica(ica_path)

        events = mne.read_events(eve_path)

        raw = ica.apply(
            raw,
            exclude=components_rej[key]
        )

        beh_run = beh.loc[(beh.run == runs[int(file_ix)])]
        freq = "time"
        low_freq, high_freq  = filter_list[freq]
        filtered = raw.copy()
        del raw # remove for Hilbert transform

        filter_picks = mne.pick_types(
            filtered.info,
            meg=True,
            ref_meg=True,
            stim=False,
            eog=True
        )

        filtered = filtered.filter(
            low_freq,
            None,
            method="fir",
            phase="minimum",
            n_jobs=-1,
            picks=filter_picks
        )
        
        filtered = filtered.filter(
            None,
            high_freq,
            method="fir",
            phase="minimum",
            n_jobs=-1,
            picks=filter_picks
        )

        onsets_p2 = mne.pick_events(
            events, include=list(np.arange(10,18))
        )
        
        onsets_p2[:, 0] -= 500 # 2s before the phase 2 trigger
        onsets_p2[:, 2] -= 10 # to match the conditions in the beh

        for ix, event in enumerate(onsets_p2):
            epoch = mne.Epochs(
                filtered,
                events=[event],
                baseline=None,
                preload=True,
                tmin=-1.5,
                tmax=5.5,
                detrend=1
            )
            epoch.apply_baseline((-0.2, 0.0))
            if int(file_ix) == 0:
                info = epoch.info 
            epoch = mne.EpochsArray(
                epoch.get_data(),
                info,
                events=np.array([event]),
                tmin=epoch.tmin
            )
            long_epochs.append(epoch)
        del filtered
    long_epochs = mne.concatenate_epochs(long_epochs, add_offset=True)

    engage = beh.engage_ix.values[:, np.newaxis] + 375
    change = beh.change_ix.values[:, np.newaxis]
    change[change == 512] = 500
    change += 375

    engage_epochs = mne.EpochsArray(
        np.array([epo[:,engage[ix][0]-250:engage[ix][0]+251] for ix, epo in enumerate(long_epochs.get_data())]),
        info=info,
        events=long_epochs.events,
        tmin=-1
    )

    change_epochs = mne.EpochsArray(
        np.array([epo[:,change[ix][0]-250:change[ix][0]+251] for ix, epo in enumerate(long_epochs.get_data())]),
        info=info,
        events=long_epochs.events,
        tmin=-1
    )

    long_epochs = long_epochs.crop(tmin=-0.5, tmax=4)

    print("beh:", np.mean(beh.conditions.values == long_epochs.events[:,2]))

    long_out_path = op.join(
        meg_subject_dir,
        "long-{}-epo.fif".format(freq)
    )

    engage_out_path = op.join(
        meg_subject_dir,
        "engage-{}-epo.fif".format(freq)
    )

    change_out_path = op.join(
        meg_subject_dir,
        "change-{}-epo.fif".format(freq)
    )

    long_epochs.save(long_out_path)
    engage_epochs.save(engage_out_path)
    change_epochs.save(change_out_path)
