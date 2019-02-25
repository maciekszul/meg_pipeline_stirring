import os
from surfer import Brain
from utilities import files
from nibabel.freesurfer import io


mri_path = "/cubric/data/c1557187/stirring_MEG/mri/fs_results"
os.environ["SUBJECTS_DIR"] = mri_path
mmp_path = "/home/c1557187/data/stirring_MEG/mri/fs_results/MMP/"

subject_id = "fsaverage"
hemi = "lh"
surf = "inflated"
subj_labels = os.path.join(
    mmp_path,
    subject_id,
    "label"
)

mmp_labels = files.get_files(subj_labels, hemi, "label", wp=True)[0]
annot_file = os.path.join(
    mri_path, 
    subject_id, 
    "label",
    "{}.{}_HCP-MMP1.annot".format(hemi, subject_id)
)
brain = Brain(subject_id, hemi, surf)
# brain.add_annotation("{}_HCP-MMP1".format(subject_id), borders=False)

# annot = io.read_annot(annot_file)
