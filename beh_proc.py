from tools import files
import numpy as np
import pandas as pd




# conditions = {
#     0: (("high", "right"), ("high", "right")),
#     1: (("high", "right"), ("high", "left")),
#     2: (("high", "right"), ("low", "right")),
#     3: (("high", "right"), ("low", "left")),
#     4: (("high", "left"), ("high", "right")),
#     5: (("high", "left"), ("high", "left")),
#     6: (("high", "left"), ("low", "right")),
#     7: (("high", "left"), ("low", "left"))
# }
# x[0] phase 1, x[1] phase 2
# x[0][0] coherence, x[0][1] direction
# high 1, low -1
# right 1, left -1
cond_dict = {
    0: ((1, 1), (1, 1)),
    1: ((1, 1), (1, -1)),
    2: ((1, 1), (-1, 1)),
    3: ((1, 1), (-1, -1)),
    4: ((1, -1), (1, 1)),
    5: ((1, -1), (1, -1)),
    6: ((1, -1), (-1, 1)),
    7: ((1, -1), (-1, -1))
}

beh_path = "/cubric/scratch/c1557187/stirring/BEH"

beh_files = files.get_files(
    beh_path,
    "",
    "matched.gz"
)[2]

beh_files.sort()


beh_file = beh_files[0]
for beh_file in beh_files:
    beh = pd.read_pickle(beh_file)
    beh = beh.sort_values(["run", "trial"])
    beh.reset_index(inplace=True, drop=True)
    beh["movement_direction_phase_1"] = beh.degs.apply(lambda x: np.mean(x[:500])/2)
    beh["movement_direction_phase_2"] = beh.degs.apply(lambda x: np.mean(x[500:])/2)
    beh["dot_motion_direction_phase_1"] = beh.conditions.apply(lambda x: cond_dict[x][0][1])
    beh["dot_motion_direction_phase_2"] = beh.conditions.apply(lambda x: cond_dict[x][1][1])
    beh["dot_coherence_phase_1"] = beh.conditions.apply(lambda x: cond_dict[x][0][0])
    beh["dot_coherence_phase_2"] = beh.conditions.apply(lambda x: cond_dict[x][1][0])

    beh.to_pickle(beh_file)
    print(beh_file)

# beh.movement_direction_phase_1.index.values[np.abs(beh.movement_direction_phase_1) > 0.5]