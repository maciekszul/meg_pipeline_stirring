import mne
import os.path as op
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from tools import files
import subprocess as sp

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

proc_path = "/home/c1557187/data/stirring_MEG/proc/02_resamp_ica/"
proc_file_raw = op.join(proc_path, subj, "resamp_ica_raw.fif")
proc_events = op.join(proc_path, subj, "resamp_raw-eve.fif")

# events info
event_id = {
            1: 'hR_hR', 
            2: 'hR_hL', 
            3: 'hR_lR', 
            4: 'hR_lL', 
            5: 'hL_hR', 
            6: 'hL_hL', 
            7: 'hL_lR', 
            8: 'hL_lL'
            }


# read in the info from raw data
info = mne.io.read_info(raw_files[0], verbose=False)

if pipeline_params["epoch"]:
    raw = proc_file_raw
    events = proc_events
    
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
        events, 
        event_id=event_id, 
        tmin=0.55, 
        tmax=4.75,
        picks=picks, 
        preload=True, 
        detrend=0, 
        baseline=(0.55,0.75),
        verbose=False
        )
    
    epochs.apply_baseline((0.55, 0.75))

if pipeline_params["coreg_viz"]:
    mne.viz.plot_alignment(
        info,
        trans_file, 
        subject=subj, 
        subjects_dir=mri_dir, 
        surfaces="head-dense", 
        dig=True
    )

if pipeline_params["bem_surf"]:
    import matplotlib.pylab as plt
    mne.viz.plot_bem(
        subject=subj,
        subjects_dir=mri_dir,
        brain_surfaces="white",
        orientation="coronal"
    )

if pipeline_params["compute_source_space"]:
    src = mne.setup_source_space(
        subject=subj, 
        subjects_dir=mri_dir, 
        spacing="ico4", 
        add_dist=False
    )

if pipeline_params["source_space_3d"]:
    from mayavi import mlab
    from surfer import Brain
    hemispheres = ["lh", "rh"]

    brain = Brain(
        subj,
        hemispheres[0],
        "inflated",
        subjects_dir=mri_dir
    )

    surf = brain.geo[hemispheres[0]]

    vertidx = np.where(src[0]["inuse"])[0]

    mlab.points3d(
        surf.x[vertidx], 
        surf.y[vertidx],
        surf.z[vertidx], 
        color=(1, 1, 0), 
        scale_factor=1.5
    )

if pipeline_params["make_fwd_solution"]:
    # conductivity = (0.3, )
    model = mne.make_bem_model(
        subject=subj,
        ico=4,
        # conductivity=conductivity,
        subjects_dir=mri_dir
    )

    bem = mne.make_bem_solution(model)
    
    for ix, fif in enumerate(raw_files):
        fwd = mne.make_forward_solution(
            fif,
            trans=trans_file,
            src=src,
            bem=bem,
            meg=True,
            eeg=False,
            mindist=5.0,
            n_jobs=-1
        )
        output_name = op.join(out_path, "{0}-fwd.fif".format(ix))
        mne.write_forward_solution(output_name, fwd)


if pipeline_params["check_fwd_solution"]:
    fwd_sol_files = files.get_files(out_path, "", "-fwd.fif")[0]
    fwd_sol_files.sort()
    sensi_list = []
    for ix, fwd_file in enumerate(fwd_sol_files):
        fwd = mne.read_forward_solution(fwd_file)
        mne.convert_forward_solution(fwd, surf_ori=True, copy=False)
        leadfield = fwd["sol"]["data"]
        mag_map = mne.sensitivity_map(fwd, ch_type="mag", mode="fixed")
        sensi_list.append(mag_map.data.ravel())
    
    plt.figure()
    plt.hist(
        sensi_list,
        bins=20,
        label=["File {}".format(i) for i in range(len(fwd_sol_files))],
    )
    plt.legend()
    plt.title("Normal orientation sensitivity, subject {}".format(subj))
    plt.xlabel("sensitivity")
    plt.ylabel("count")
    fig_file = op.join(out_path, "orientation_sensitivity_all_files.png")
    plt.savefig(fig_file)


if pipeline_params["check_fwd_solution_3d_file"][0]:
    fwd_sol_files = files.get_files(out_path, "", "-fwd.fif")[0]
    fwd_sol_files.sort()
    ix = pipeline_params["check_fwd_solution_3d_file"][1]
    fwd_file = fwd_sol_files[ix]
    fwd = mne.read_forward_solution(fwd_file)
    mne.convert_forward_solution(fwd, surf_ori=True, copy=False)
    leadfield = fwd["sol"]["data"]
    mag_map = mne.sensitivity_map(fwd, ch_type="mag", mode="fixed")
    mag_map.plot(
        time_label="Magnetometer sensitivity",
        subjects_dir=mri_dir,
        clim=dict(lims=[0, 50, 100])
    )


if pipeline_params["convert_noise"]:
    raw_ds = files.get_folders_files(raw_dir)[0]
    noise_ds = files.items_cont_str(raw_ds, "Noise", sort=True)[0]
    noise_fif = op.join(raw_dir, "noise-raw.fif")
    ctf2fif = "/cubric/data/c1557187/MNE/bin/mne_ctf2fiff"
    sp.call(
        [ctf2fif,
        "--ds",
        noise_ds,
        "--fif",
        noise_fif]
    )


if pipeline_params["process_noise"]:
    cov_file = files.get_files(raw_dir, "", "noise-raw.fif")[0][0]
    cov_raw = mne.io.read_raw_fif(cov_file, preload=True, verbose=False)
    cov_raw.filter(
        1, 
        90, 
        n_jobs=-1, 
        filter_length='auto',
        fir_design='firwin',
        method='fir'
    )

    cov_out_file = op.join(
        out_path,
        "processed-noise.fif"
    )

    cov_raw.save(
        cov_out_file, 
        fmt='single', 
        split_size='2GB'
    )


if pipeline_params["compute_raw_covariance"][0]:
    cov_file = files.get_files(raw_dir, "", "noise-raw.fif")[0][0]
    cov_raw = mne.io.read_raw_fif(cov_file, preload=False, verbose=False)
    cov = mne.compute_raw_covariance(cov_raw, method="auto", rank=None)
    cov.plot(cov_raw.info, proj=True)
