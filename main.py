import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
from data_reader import ReadData
from data_formatter import DataFormatter
from params import Params

# %%
params = Params()

nexdata = ReadData(params.filename, os.getcwd() + params.directory)
data = DataFormatter(nexdata.raw_data, params.well_events, params.eating_events, params.colors)
data_trials = data.trials

# %%

def fft(spike_times: np.ndarray):
    """
    Calculate the FFT of spike times
    Parameters:
    ___________
    spike_times: array
        Array of spike times
    Returns:
    ________
    freqs: array
        Array of frequencies
    spike_fft: array
        Array of FFT output
    """
    # Normalize the spike times by subtracting the first spike time
    spike_times = np.asarray(spike_times)
    spike_times = spike_times - spike_times[0]
    # spike_times = spike_times[:100]
    time_intervals = np.diff(spike_times)
    N = len(spike_times)
    sampling_frequency = 1000
    nyquist_frequency = int(sampling_frequency / 2)

    Hz = np.linspace(0, int(nyquist_frequency), int(np.floor(N / 2) + 1))
    max_frequency = 1 / (2 * np.median(time_intervals))

    if max_frequency > nyquist_frequency:
        print("Warning: time intervals are too small compared to sampling frequency for FFT")
        return None, None
    else:
        print("Time intervals are suitable for FFT")

    # Pad the spike times array with zeros to the nearest power of two
    n = int(2 ** np.ceil(np.log2(len(spike_times))))
    spike_times_padded = np.zeros(n)
    spike_times_padded[:len(spike_times)] = spike_times

    # Calculate the FFT of the padded spike times array
    spike_fft = np.fft.fft(spike_times_padded)
    freqs = np.fft.fftfreq(len(spike_times_padded), d=1 / sampling_frequency)

    plt.plot(freqs, np.abs(spike_fft))
    plt.ylabel('Amplitude')
    plt.xlabel('Freq - Hz')
    plt.show()

    return freqs, spike_fft

freqs, spike_fft = fft(data_trials['e_BR'][1].neuron)
plt.plot(freqs, np.abs(spike_fft))
plt.ylabel('Amplitude')
plt.xlabel('Freq - Hz')
plt.show()

# # Plot the FFT output
# plt.plot(freqs, np.abs(spike_fft))
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.show()



# df1 = data.df_sorted
# df_e = df1[df1['event'].isin(eating_events)]
# df_s = df1[df1['event'].isin(['Spont', 'spont'])]
# df_w = df1[(df1.index <= df_e.index[0]) & (df1['event'].isin(well_events))]
#
# uni = data.df_sorted.loc[data.df_sorted.mid.drop_duplicates().index, ['neuron', 'color', 'counts', 'mid']]
#
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






