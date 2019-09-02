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


L_CH_CWS= []
L_CH_ACW= []
L_NCH_CWS = []
L_NCH_ACW = []
E_CH_CWS= []
E_CH_ACW= []
E_NCH_CWS = []
E_NCH_ACW = []
CH_CH_CWS= []
CH_CH_ACW= []
CH_NCH_CWS = []
CH_NCH_ACW = []


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
    
    # CLOCKWISE (-1) vs ANTICLOCKWISE (1)
    no_change_cws_ix = beh.index.values[
        ((beh.binned_cond == 2) |
        (beh.binned_cond == 0)) &
        (beh.movement_direction_phase_2 < -0.5)
    ]

    no_change_acw_ix = beh.index.values[
        ((beh.binned_cond == 2) |
        (beh.binned_cond == 0)) &
        (beh.movement_direction_phase_2 > 0.5)
    ]

    L_NCH_CWS.append(np.mean(long[no_change_cws_ix], axis=0))
    L_NCH_ACW.append(np.mean(long[no_change_acw_ix], axis=0))
    E_NCH_CWS.append(np.mean(engage[no_change_cws_ix], axis=0))
    E_NCH_ACW.append(np.mean(engage[no_change_acw_ix], axis=0))
    CH_NCH_CWS.append(np.mean(change[no_change_cws_ix], axis=0))
    CH_NCH_ACW.append(np.mean(change[no_change_acw_ix], axis=0))

    change_cws_ix = beh.index.values[
        ((beh.binned_cond == 3) |
        (beh.binned_cond == 1)) &
        (beh.movement_direction_phase_2 < -0.5)
    ]

    change_acw_ix = beh.index.values[
        ((beh.binned_cond == 3) |
        (beh.binned_cond == 1)) &
        (beh.movement_direction_phase_2 > 0.5)
    ]

    L_CH_CWS.append(np.mean(long[change_cws_ix], axis=0))
    L_CH_ACW.append(np.mean(long[change_acw_ix], axis=0))
    E_CH_CWS.append(np.mean(engage[change_cws_ix], axis=0))
    E_CH_ACW.append(np.mean(engage[change_acw_ix], axis=0))
    CH_CH_CWS.append(np.mean(change[change_cws_ix], axis=0))
    CH_CH_ACW.append(np.mean(change[change_acw_ix], axis=0))


cws_v_acw = {
    "Long": {
        "No change of direction": {
            "clockwise": L_NCH_CWS,
            "anti-clockwise": L_NCH_ACW
        },
        "Change of direction": {
            "clockwise": L_CH_CWS,
            "anti-clockwise": L_CH_ACW
        }
    },
    "Engage": {
        "No change of direction": {
            "clockwise": E_NCH_CWS,
            "anti-clockwise": E_NCH_ACW
        },
        "Change of direction": {
            "clockwise": E_CH_CWS,
            "anti-clockwise": E_CH_ACW
        }
    },
    "Change": {
        "No change of direction": {
            "clockwise": CH_NCH_CWS,
            "anti-clockwise": CH_NCH_ACW
        },
        "Change of direction": {
            "clockwise": CH_CH_CWS,
            "anti-clockwise": CH_CH_ACW
        }
    }
}


evd_file = op.join(
    output_path,
    "clockwise_anticlocwise_all_sensors_mean.npy"
)
np.save(evd_file, cws_v_acw)
