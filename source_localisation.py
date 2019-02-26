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

events_ids = {
              '1hR_hR': 1, 
              '1hR_hL': 2, 
              '1hR_lR': 3, 
              '1hR_lL': 4, 
              '1hL_hR': 5, 
              '1hL_hL': 6, 
              '1hL_lR': 7, 
              '1hL_lL': 8, 
              '2hR_hR': 10, 
              '2hR_hL': 11, 
              '2hR_lR': 12, 
              '2hR_lL': 13, 
              '2hL_hR': 14, 
              '2hL_hL': 15, 
              '2hL_lR': 16, 
              '2hL_lL': 17, 
              'blank': 64
              }

# read in the info from raw data
info = mne.io.read_info(raw_files[0], verbose=False)

if pipeline_params["epoch_4s"]:
    raw = mne.io.read_raw_fif(
        proc_file_raw, 
        preload=True,
        verbose=False
    )
    events = mne.read_events(
        proc_events
    )
    
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
        event_id=list(range(1, 9)), 
        tmin=0.55, 
        tmax=4.75,
        picks=picks, 
        preload=True, 
        detrend=0, 
        baseline=(0.55,0.75),
        verbose=False
        )
    
    epochs.apply_baseline((0.55, 0.75))

    epochs_file_out = op.join(
        out_path,
        "ica_1_90_4s-epo.fif"
    )

    epochs.save(
        epochs_file_out,
        split_size="2GB",
        fmt="single",
        verbose=False
    )


if pipeline_params["epoch_2s"]:
    raw = mne.io.read_raw_fif(
        proc_file_raw, 
        preload=True,
        verbose=False
    )
    events = mne.read_events(
        proc_events
    )
    
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
        events, 
        event_id=list(range(1, 9)), 
        tmin=0.55, 
        tmax=2.75,
        picks=picks, 
        preload=True, 
        detrend=0, 
        baseline=(0.55,0.75),
        verbose=False
        )
    
    epochs1.apply_baseline((0.55, 0.75))

    epochs2 = mne.Epochs(
        raw,
        events,
        event_id=list((range(10, 18))),
        tmin=2.55,
        tmax=4.75,
        picks=picks,
        preload=True,
        detrend=0,
        baseline=(2.55, 2.75),
        verbose=False
    )

    epochs2.apply_baseline((2.55, 2.75))

    epochs_file_out1 = op.join(
        out_path,
        "ica_1_90_phase1-epo.fif"
    )

    epochs_file_out2 = op.join(
        out_path,
        "ica_1_90_phase2-epo.fif"
    )

    epochs1.save(
        epochs_file_out1,
        split_size="2GB",
        fmt="single",
        verbose=False
    )

    epochs1.save(
        epochs_file_out2,
        split_size="2GB",
        fmt="single",
        verbose=False
    )


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
        spacing="ico5", 
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


if pipeline_params["make_bem_model"]:
    conductivity = (0.3, )
    model = mne.make_bem_model(
        subject=subj,
        ico=5,
        conductivity=conductivity,
        subjects_dir=mri_dir
    )

    bem = mne.make_bem_solution(model)



if pipeline_params["make_fwd_solution"]:
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


if pipeline_params["make_fwd_solution_epo"]:
    epochs = files.get_files(out_path, "", "4s-epo.fif", wp=True)[0][0]
    
    fwd = mne.make_forward_solution(
        epochs,
        trans=trans_file,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        mindist=5.0,
        n_jobs=-1
    )
    output_name = op.join(out_path, "epochs-fwd.fif")
    mne.write_forward_solution(output_name, fwd, overwrite=True)


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


if pipeline_params["room_noise"]:
    cov_file = files.get_files(
        raw_dir, 
        "", 
        "noise-raw.fif"
    )[0][0]
    cov_raw = mne.io.read_raw_fif(
        cov_file, 
        preload=True, 
        verbose=False
    )
    cov_raw.resample(250, npad="auto")
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
        "room-noise.fif"
    )

    cov_raw.save(
        cov_out_file, 
        fmt='single', 
        split_size='2GB'
    )

if pipeline_params["rs_noise"]:
    cov_file = files.get_files(
        raw_dir, 
        "", 
        "rs-raw.fif"
    )[0][0]
    cov_raw = mne.io.read_raw_fif(
        cov_file, 
        preload=True, 
        verbose=False
    )
    events = mne.find_events(cov_raw, stim_channel='UPPT001')
    cov_raw, events = cov_raw.resample(250, npad="auto", events=events)
    tmin, tmax = events[:,0]/250
    cov_raw = cov_raw.crop(tmin=tmin, tmax=tmax)
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
        "rs-noise.fif"
    )

    cov_raw.save(
        cov_out_file, 
        fmt='single', 
        split_size='2GB'
    )


if pipeline_params["compute_noise_covariance"][0]:
    cov_file = files.get_files(
        out_path,
        "",
        "{}-noise.fif".format(pipeline_params["compute_noise_covariance"][1])
    )[0][0]
    cov_raw = mne.io.read_raw_fif(
        cov_file, 
        preload=False, 
        verbose=False
    )
    picks = mne.pick_types(
        cov_raw.info, 
        meg=True, 
        eeg=False, 
        stim=False, 
        eog=False, 
        ref_meg='auto', 
        exclude='bads'
    ) 
    noise_cov = mne.compute_raw_covariance(
        cov_raw, 
        method=['shrunk', 'empirical'], 
        rank=None,
        picks=picks,
        n_jobs=-1
    )
    

if pipeline_params["compute_inverse_operator"]:
    epochs = mne.read_epochs(
        files.get_files(out_path, "", "4s-epo.fif")[0][0],
        preload=False,
        verbose=False
    )
    
    fwd = mne.read_forward_solution(
        files.get_files(out_path, "", "epochs-fwd.fif")[0][0],
        verbose=False
    )

    inverse_operator = mne.minimum_norm.make_inverse_operator(
        epochs.info,
        fwd,
        noise_cov,
        loose=0.2,
        depth=0.8
    )

    inverse_operator_file_path = op.join(
        out_path,
        "{}-inv.fif".format(pipeline_params["compute_noise_covariance"][1])
    )

    mne.minimum_norm.write_inverse_operator(
        inverse_operator_file_path,
        inverse_operator
    )


if pipeline_params["compute_inverse_solution"][0]:
    epochs = mne.read_epochs(
        files.get_files(out_path, "", "4s-epo.fif")[0][0],
        preload=False,
        verbose=False
    )

    inverse_operator = mne.minimum_norm.read_inverse_operator(
        files.get_files(out_path, "rs", "-inv.fif")[0][0],
        verbose=False
    )

    method_dict = {
        "dSPM": (8, 12, 15),
        "sLORETA": (3, 5, 7),
        "eLORETA": (0.75, 1.25, 1.75)
    }
    method = pipeline_params["compute_inverse_solution"][1]
    snr = 3.
    lambda2 = 1. / snr ** 2
    lims = method_dict[method]

    stc = mne.minimum_norm.apply_inverse_epochs(
        epochs,
        inverse_operator,
        lambda2,
        method=method,
        pick_ori=None,
        verbose=True
    )

    # save stc + visualisation
