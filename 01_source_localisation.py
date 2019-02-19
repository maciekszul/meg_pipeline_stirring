import mne
import os.path as op
import argparse
import json
import numpy as np
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

# read in the info from raw data
info = mne.io.read_info(raw_files[0])

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
        spacing="ico3", 
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