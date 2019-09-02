import mne
import pandas as pd
import numpy as np
from tools import files
import os.path as op
from scipy.stats import trim_mean, sem
from tqdm import tqdm

output_path = "/cubric/scratch/c1557187/stirring/RESULTS/ERF"

beh_path = "/cubric/scratch/c1557187/stirring/BEH"
beh_files = files.get_files(
    beh_path,
    "",
    "matched.gz"
)[2]

beh_files.sort()

epo_path = "/cubric/scratch/c1557187/stirring/MEG"
subjects = files.get_folders_files(
    epo_path,
    wp=False
)[0]
subjects.sort()

long_files = [files.get_files(op.join(epo_path, i), "long", "epo.fif")[2][0] for i in subjects]
long_files.sort()

engage_files = [files.get_files(op.join(epo_path, i), "engage", "epo.fif")[2][0] for i in subjects]
engage_files.sort()

change_files = [files.get_files(op.join(epo_path, i), "change", "epo.fif")[2][0] for i in subjects]
change_files.sort()

main_info = mne.read_epochs(long_files[0], preload=True).pick_types(ref_meg=False, eog=False, stim=False).info

sensor_groupings = {
    "Left Occipital": [i for i in main_info["ch_names"] if "MLO" in i],
    "Right Occipital": [i for i in main_info["ch_names"] if "MRO" in i],
    "All Occiptal": [i for i in main_info["ch_names"] if "O" in i],
    "Left Central": [i for i in main_info["ch_names"] if "MLC" in i],
    "Right Central": [i for i in main_info["ch_names"] if "MRC" in i],
    "All Central": [i for i in main_info["ch_names"] if "C" in i],
    "Left Parietal": [i for i in main_info["ch_names"] if "MLP" in i],
    "Right Parietal": [i for i in main_info["ch_names"] if "MRP" in i],
    "All Parietal": [i for i in main_info["ch_names"] if "P" in i],
    "Left Frontal": [i for i in main_info["ch_names"] if "MLF" in i],
    "Right Frontal": [i for i in main_info["ch_names"] if "MRF" in i],
    "All Frontal": [i for i in main_info["ch_names"] if "F" in i],
    "All Sensors": main_info["ch_names"]
}

bin_cond = {
    0: "no change of coherence and direction",
    1: "change of direction only",
    2: "change of coherence but no direction",
    3: "change of both coherence and direction"
}

# iter_ = [list(zip(beh_files, long_files, engage_files, change_files))[1]]
iter_ = zip(beh_files, long_files, engage_files, change_files)


L_CH_E = []
L_CH_D = []
L_NCH_E = []
L_NCH_D = []
E_CH_E = []
E_CH_D = []
E_NCH_E = []
E_NCH_D = []
CH_CH_E = []
CH_CH_D = []
CH_NCH_E = []
CH_NCH_D = []

for beh_path, long_path, engage_path, change_path in tqdm(iter_):
    beh = pd.read_pickle(beh_path)
    long = mne.read_epochs(long_path, preload=True).pick_types(
        ref_meg=False, 
        eog=False, 
        stim=False
    ).get_data()
    engage = mne.read_epochs(engage_path, preload=True).pick_types(
        ref_meg=False, 
        eog=False, 
        stim=False
    ).get_data()
    change = mne.read_epochs(change_path, preload=True).pick_types(
        ref_meg=False, 
        eog=False, 
        stim=False
    ).get_data()
    
    # EASY (-1) vs DIFFICULT (1)
    no_change_easy_ix = beh.index.values[
        (beh.binned_cond == 0)
    ]
    no_change_diff_ix = beh.index.values[
        (beh.binned_cond == 2)
    ]
    L_NCH_E.append(np.mean(long[no_change_easy_ix], axis=0))
    L_NCH_D.append(np.mean(long[no_change_diff_ix], axis=0))
    E_NCH_E.append(np.mean(engage[no_change_easy_ix], axis=0))
    E_NCH_D.append(np.mean(engage[no_change_diff_ix], axis=0))
    CH_NCH_E.append(np.mean(change[no_change_easy_ix], axis=0))
    CH_NCH_D.append(np.mean(change[no_change_diff_ix], axis=0))

    change_easy_ix = beh.index.values[
        (beh.binned_cond == 1)
    ]
    change_diff_ix = beh.index.values[
        (beh.binned_cond == 3)
    ]

    L_CH_E.append(np.mean(long[change_easy_ix], axis=0))
    L_CH_D.append(np.mean(long[change_diff_ix], axis=0))
    E_CH_E.append(np.mean(engage[change_easy_ix], axis=0))
    E_CH_D.append(np.mean(engage[change_diff_ix], axis=0))
    CH_CH_E.append(np.mean(change[change_easy_ix], axis=0))
    CH_CH_D.append(np.mean(change[change_diff_ix], axis=0))


easy_v_diff = {
    "Long": {
        "No change of direction": {
            "easy": L_NCH_E,
            "difficult": L_NCH_D
        },
        "Change of direction": {
            "easy": L_CH_E,
            "difficult": L_CH_D
        }
    },
    "Engage": {
        "No change of direction": {
            "easy": E_NCH_E,
            "difficult": E_NCH_D
        },
        "Change of direction": {
            "easy": E_CH_E,
            "difficult": E_CH_D
        }
    },
    "Change": {
        "No change of direction": {
            "easy": CH_NCH_E,
            "difficult": CH_NCH_D
        },
        "Change of direction": {
            "easy": CH_CH_E,
            "difficult": CH_CH_D
        }
    }
}




evd_file = op.join(
    output_path,
    "easy_difficult_all_sensors_mean.npy"
)
np.save(evd_file, easy_v_diff)
