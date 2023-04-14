import numpy as np
import matplotlib.pyplot as plt

# Generate a random array of spike times
spike_times = np.random.uniform(0, 10, 100)

# Sort the spike times
spike_times = np.sort(spike_times)

# Calculate the time intervals between spikes
time_intervals = np.diff(spike_times)

# Calculate the sampling frequency from the median time interval
sampling_frequency = 1 / np.median(time_intervals)

# Pad the spike times array with zeros to the nearest power of two
n = int(2 ** np.ceil(np.log2(len(spike_times))))
spike_times_padded = np.zeros(n)
spike_times_padded[:len(spike_times)] = spike_times

# Calculate the FFT of the padded spike times array
spike_fft = np.fft.fft(spike_times_padded)

# Calculate the frequency values for the FFT output
freqs = np.fft.fftfreq(len(spike_times_padded)) * sampling_frequency

# Plot the FFT output

plt.plot(freqs, np.abs(spike_fft))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()