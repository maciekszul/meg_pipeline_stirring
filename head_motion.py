import mne
import os.path as op
import numpy as np
import json
from tools import files
import pandas as pd
import matplotlib.pylab as plt

path = "/home/c1557187/data/stirring_MEG/raw/001/050516-41_Decision_20170403_01.ds"

raw = mne.io.read_raw_ctf(path, preload=True)

xyz_data = {
    'nasion': np.vstack([
        raw.get_data([310][0]), 
        raw.get_data([311][0]), 
        raw.get_data([312][0])
    ]),
    'left_coil': np.vstack([
        raw.get_data([313][0]),
        raw.get_data([314][0]),
        raw.get_data([315][0])
    ]),
    'right_coil': np.vstack([
        raw.get_data([316][0]),
        raw.get_data([317][0]),
        raw.get_data([318][0])
    ])
}
# time = np.arange(len(xyz_data["nasion"][2])) / 1200
# plt.plot(time, (xyz_data["nasion"][2]-xyz_data["nasion"][2][0])*1000)
# plt.plot(time, (xyz_data["left_coil"][2]-xyz_data["left_coil"][2][0])*1000)
# plt.plot(time, (xyz_data["right_coil"][2]-xyz_data["right_coil"][2][0])*1000)
# plt.show()

n = xyz_data["nasion"].transpose()
l = xyz_data["left_coil"].transpose()
r = xyz_data["right_coil"].transpose()
