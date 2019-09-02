import mne
import pandas as pd
import numpy as np
from tools import files
import os.path as op
import matplotlib.pylab as plt
from matplotlib import gridspec
from scipy.stats import trim_mean, sem
from mne.stats import permutation_cluster_test
from mne.baseline import rescale

output_path = "/cubric/scratch/c1557187/stirring/RESULTS/ERF/VIZ"

epo_file = "/cubric/scratch/c1557187/stirring/MEG/001/change-time-epo.fif"
main_info = mne.read_epochs(epo_file, preload=True).pick_types(ref_meg=False, eog=False, stim=False).info

data_file = "/cubric/scratch/c1557187/stirring/RESULTS/ERF/easy_difficult_all_sensors_mean.npy"

data = np.load(data_file).item()

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
    "All Frontal": [i for i in main_info["ch_names"] if "F" in i]
}

# subset = "Left Occipital"
# selection = "Change of direction"
selection = "No change of direction"

mask_params = dict(
    marker='o', 
    markerfacecolor='w', 
    markeredgecolor='r',
    linewidth=0, 
    markersize=4
)

long_times = np.linspace(-0.5, 4, num=1126)
short_times = np.linspace(-1, 1, num=501)

y_lims = (-10,10)

col = {
    "easy": "#EB2188",
    "difficult": "#080A52"
}

sign_colour = "#00ff00"
non_sign_colour = "#cccccc"
for subset in sensor_groupings.keys():
    ch_ix = mne.pick_channels(
        main_info["ch_names"],
        sensor_groupings[subset]
    )

    gs = gridspec.GridSpec(
        2, 
        4, 
        wspace=0.2, 
        hspace=0.3, 
        width_ratios=[1, 1, 1, 1],
        height_ratios=[2, 2]
        )
    fig = plt.figure(figsize=(10, 8))

    sens_grp = fig.add_subplot(gs[0,0])
    dummy_data = np.zeros(271)
    ch_selection = np.zeros(271, dtype=bool)
    ch_selection[mne.pick_channels(main_info["ch_names"], sensor_groupings[subset])] = True
    mne.viz.plot_topomap(
        dummy_data,
        main_info,
        cmap="Greys",
        vmin=0,
        vmax=0,
        mask=ch_selection,
        mask_params=mask_params,
        axes=sens_grp,
        show=False
    )
    sens_grp.title.set_text("{} Sensors".format(subset))

    long = fig.add_subplot(gs[0, 1:])
    # data proc
    easy = np.array(data["Long"][selection]["easy"]) * 1e14
    easy = np.mean(easy[:, ch_ix, :], axis=1)
    easy_mean = np.mean(easy, axis=0)
    easy_sem = sem(easy, axis=0)
    difficult = np.array(data["Long"][selection]["difficult"]) * 1e14
    difficult = np.mean(difficult[:, ch_ix, :], axis=1)
    difficult_mean = np.mean(difficult, axis=0)
    difficult_sem = sem(difficult, axis=0)

    threshold=2.0
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        [easy, difficult], 
        n_permutations=5000, 
        threshold=threshold, 
        tail=0, 
        n_jobs=-1
    )

    for i_c, c in enumerate(clusters):
        c = c[0]
        if cluster_p_values[i_c] < 0.05:
            long.axvspan(
                long_times[c.start], 
                long_times[c.stop-1],
                color=sign_colour,
                alpha=0.2
            )
        elif cluster_p_values[i_c] < 0.5:
            long.axvspan(
                long_times[c.start], 
                long_times[c.stop-1],
                color=non_sign_colour,
                alpha=0.2
            )

    # end data proc

    long.plot(long_times, easy_mean, linewidth=1, color=col["easy"], label="No change of coherence")
    long.fill_between(
        long_times,
        easy_mean - easy_sem,
        easy_mean + easy_sem,
        color=col["easy"], 
        alpha=0.2, 
        linewidth=0
    )

    long.plot(long_times, difficult_mean, linewidth=1, color=col["difficult"], label="Change of coherence")
    long.fill_between(
        long_times,
        difficult_mean - difficult_sem,
        difficult_mean + difficult_sem,
        color=col["difficult"], 
        alpha=0.2, 
        linewidth=0
    )
    long.legend(loc=4)

    long.title.set_text('Full duration of the trial')
    long.yaxis.set_label_position("right")
    long.yaxis.tick_right()
    long.set_ylabel("fT")
    long.set_xlabel("Time [s]")
    long.set_xlim((-0.5, 4))
    long.set_ylim(y_lims)
    long_xticks = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    long.set_xticks(long_xticks)
    long.set_xticklabels([str(i) for i in long_xticks])
    long.axvspan(1.95, 2.05, facecolor="#000000", alpha=0.1)
    long.axhline(0, linewidth=0.5, color="#000000")
    long.axvline(0, linestyle="--", linewidth=0.5, color="#000000")

    engage = fig.add_subplot(gs[1, :2])
    # data proc
    easy = np.array(data["Engage"][selection]["easy"]) * 1e14
    easy = np.mean(easy[:, ch_ix, :], axis=1)
    easy_mean = np.mean(easy, axis=0)
    easy_sem = sem(easy, axis=0)
    difficult = np.array(data["Engage"][selection]["difficult"]) * 1e14
    difficult = np.mean(difficult[:, ch_ix, :], axis=1)
    difficult_mean = np.mean(difficult, axis=0)
    difficult_sem = sem(difficult, axis=0)

    threshold=2.0
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        [easy, difficult], 
        n_permutations=5000, 
        threshold=threshold, 
        tail=0, 
        n_jobs=-1
    )

    for i_c, c in enumerate(clusters):
        c = c[0]
        if cluster_p_values[i_c] < 0.05:
            engage.axvspan(
                short_times[c.start], 
                short_times[c.stop-1],
                color=sign_colour,
                alpha=0.2
            )
        elif cluster_p_values[i_c] < 0.5:
            engage.axvspan(
                short_times[c.start], 
                short_times[c.stop-1],
                color=non_sign_colour,
                alpha=0.2
            )

    # end data proc

    engage.plot(short_times, easy_mean, linewidth=1, color=col["easy"], label="No change of coherence")
    engage.fill_between(
        short_times,
        easy_mean - easy_sem,
        easy_mean + easy_sem,
        color=col["easy"], 
        alpha=0.2, 
        linewidth=0
    )

    engage.plot(short_times, difficult_mean, linewidth=1, color=col["difficult"], label="Change of coherence")
    engage.fill_between(
        short_times,
        difficult_mean - difficult_sem,
        difficult_mean + difficult_sem,
        color=col["difficult"], 
        alpha=0.2, 
        linewidth=0
    )
    engage.legend(loc=2)

    # engage.title.set_text('Signal aligned to\nthe movement engagement')
    engage.title.set_text('Signal aligned to\nthe trial onset')
    engage.set_ylabel("fT")
    engage.set_xlabel("Time [s]")
    engage.set_xlim((-1, 1))
    engage.set_ylim(y_lims)
    engage_xticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    engage.set_xticks(engage_xticks)
    engage.set_xticklabels([str(i) for i in engage_xticks])
    engage.axhline(0, linewidth=0.5, color="#000000")
    engage.axvline(0, linestyle="--", linewidth=0.5, color="#000000")

    change = fig.add_subplot(gs[1, 2:])
    # data proc
    easy = np.array(data["Change"][selection]["easy"]) * 1e14
    easy = np.mean(easy[:, ch_ix, :], axis=1)
    easy_mean = np.mean(easy, axis=0)
    easy_sem = sem(easy, axis=0)
    difficult = np.array(data["Change"][selection]["difficult"]) * 1e14
    difficult = np.mean(difficult[:, ch_ix, :], axis=1)
    difficult_mean = np.mean(difficult, axis=0)
    difficult_sem = sem(difficult, axis=0)

    threshold=2.0
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        [easy, difficult], 
        n_permutations=5000, 
        threshold=threshold, 
        tail=0, 
        n_jobs=-1
    )

    for i_c, c in enumerate(clusters):
        c = c[0]
        if cluster_p_values[i_c] < 0.05:
            change.axvspan(
                short_times[c.start], 
                short_times[c.stop-1],
                color=sign_colour,
                alpha=0.2
            )
        elif cluster_p_values[i_c] < 0.5:
            change.axvspan(
                short_times[c.start], 
                short_times[c.stop-1],
                color=non_sign_colour,
                alpha=0.2
            )

    # end data proc

    change.plot(short_times, easy_mean, linewidth=1, color=col["easy"], label="No change of coherence")
    change.fill_between(
        short_times,
        easy_mean - easy_sem,
        easy_mean + easy_sem,
        color=col["easy"], 
        alpha=0.2, 
        linewidth=0
    )

    change.plot(short_times, difficult_mean, linewidth=1, color=col["difficult"], label="Change of coherence")
    change.fill_between(
        short_times,
        difficult_mean - difficult_sem,
        difficult_mean + difficult_sem,
        color=col["difficult"], 
        alpha=0.2, 
        linewidth=0
    )
    change.legend(loc=2)

    change.title.set_text('Signal aligned to the change\nof the movemement direction')
    change.yaxis.set_label_position("right")
    change.yaxis.tick_right()
    change.set_ylabel("fT")
    change.set_xlabel("Time [s]")
    change.set_xlim((-1, 1))
    change.set_ylim(y_lims)
    change_xticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    change.set_xticks(change_xticks)
    change.set_xticklabels([str(i) for i in change_xticks])
    change.axhline(0, linewidth=0.5, color="#000000")
    change.axvline(0, linestyle="--", linewidth=0.5, color="#000000")

    # plt.tight_layout(w_pad=0.2, h_pad=0.2)

    png = op.join(output_path, "{}_{}.png".format(selection.lower().replace(" ", "_"), subset.lower().replace(" ", "_")))
    svg = op.join(output_path, "{}_{}.svg".format(selection.lower().replace(" ", "_"), subset.lower().replace(" ", "_")))
    plt.savefig(png)
    plt.savefig(svg)

    plt.close()