
# %% PLOT TRIALS -----------------------------------------------------------------

# x=0
# well_times, eating_times, spont_times = [], [], []
# df1 = pd.DataFrame(columns=['isi', 'mid'])
# for df, wt, et, st in data.generate_trials_loose():
#
#     plot.isi_histogram(df.neuron, df.color)
#     plot.trial_histogram(df)
#     avg, std = calculate_window_stats(np.diff(df.neuron))
#     fig, ax = plt.subplots()
#     ax.plot(df.reset_index(drop=True).index[:-1], std, color='black')
#     ax.set_title("St. Dev. of Sequential ISIs \n"
#                  "(sliding window of 10 intervals)", fontweight='bold')
#     ax.set_xlabel("ISI number", fontweight='bold')
#     ax.set_ylabel("st. dev. (# of spikes)", fontweight='bold')
#     ax.set_frame_on(False)
#     ax.axvspan(
#         et[0],
#         et[-1],
#         alpha=0.05,
#         color='red'
#     )
#     ax.axvspan(
#         wt[0],
#         wt[-1],
#         alpha=0.05,
#         color='blue'
#     )
#     if st.size > 0:
#         ax.axvspan(
#             st[0],
#             st[-1],
#             alpha=0.05,
#             color='gray'
#         )
#     plt.tight_layout()
#     plt.show()

# %% PLOT ISI'S FOR ALL TRIALS ------------------------------------------------

# isi = np.diff(df_e.neuron)
# bins = np.arange(-.005, 0.2, 0.001)
# counts, _ = np.histogram(isi, bins)
# prob = counts / len(isi)
# fig, ax = plt.subplots()
# ax.bar(bins[:-1], prob, width=1e-3)
# ax.set_xlabel("ISI [s]")
# ax.set_ylabel("Frequency", fontweight='bold')
# ax.set_frame_on(False)
# plt.tight_layout()
# plt.show()
# for event, trial in data_trials.items():
#     if trial:
#         for trial_num, this_trial_df in data_trials[event].items():
#         # data_trials[event][trial]
#             single_histo(this_trial_df, 0)

# df_all_ev = pd.concat([df_spont, df_all_e, df_all_w], axis=0)
# df_all_ev = df_all_ev.drop('mid', axis=1).reset_index(drop=True)
# df_all_ev = df_all_ev.reset_index(drop=True)
#
# fig, ax = plt.subplots()
# ax.bar(df_all_ev.index, gaussian_filter(df_all_ev.counts, sigma=2), width=0.1, color=df_all_ev.color)
# ax.set_title("Everything", fontweight='bold')
# ax.axes.get_xaxis().set_visible(False)
# ax.set_ylabel("Frequency (hz)", fontweight='bold')
# ax.set_frame_on(False)
# # plt.tight_layout()
# plt.savefig(r'C:\Users\Flynn\Dropbox\Lab\fig')

# %% DATA TO MATLAB ----------------------------------------------------------

# final = {}
#
# for k, v in spike_times_dict.items(): # k = e_BR, v = {3: array, 4: array}
#     for outer, inner in v.items(): # outer = 3, inner = array
#         for k2, v2 in inner.items():
#
#         mat_structs.__setattr__(outer, inner)
#     final[k] = mat_structs
#

# scipy.io.savemat(savepath + 'from_python.mat', final)