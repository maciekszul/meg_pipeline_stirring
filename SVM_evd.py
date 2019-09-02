import sys
import mne
from mne.baseline import rescale
from tools import files
import numpy as np
import pandas as pd
import os.path as op
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from mne.decoding import (
    SlidingEstimator, 
    GeneralizingEstimator,
    cross_val_multiscore, 
    get_coef
)

try:
    id_index = int(sys.argv[1])
except:
    print("incorrect arguments")
    sys.exit()

try:
    file_index = int(sys.argv[1])
except:
    print("incorrect arguments")
    sys.exit()

output_path = "/cubric/scratch/c1557187/stirring/RESULTS/SVC/data"

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

files = [long_files, engage_files, change_files][file_index]
data_file = files[id_index]
beh_file = beh_files[id_index]

beh = pd.read_pickle(beh_file)
data = mne.read_epochs(data_file, preload=True).pick_types(
    ref_meg=False, 
    eog=False, 
    stim=False
).get_data()

labels = beh.dot_coherence_phase_2.values


# parameters for the classification
k_folds = 10 # cv folds
var_exp = 0.99  # percentage of variance

# generate iterator for cross validation
kf = StratifiedKFold(n_splits=k_folds, shuffle=True)
cv_iter = kf.split(np.zeros(data.shape), labels)

# # pipeline for classification
# cl = make_pipeline(
#     RobustScaler(), 
#     PCA(n_components=var_exp), 
#     LinearSVC(max_iter=10000, dual=False, penalty="l1")
# )

# # temporal generalisation
# temp_genr = GeneralizingEstimator(
#     cl, 
#     n_jobs=1, 
#     scoring="roc_auc"
# )

# # cross validation
# scores = cross_val_multiscore(temp_genr, data, labels, cv=cv_iter, n_jobs=-1)


# scores_path = op.join(
#     output_path,
#     "evd-{}.npy".format(subject)
# )

# np.save(scores_path, scores)

# print("saved", scores_path)