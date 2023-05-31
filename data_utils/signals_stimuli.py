from __future__ import annotations
import warnings
import sys
import ruptures as rpt
import numpy as np
import scipy.stats as stats
from scipy import signal
import pandas as pd
import math
import random
from params import Params
from .file_handling import parse_filename

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


class StimuliSignals:
    def __init__(self, nexdata: dict, filename: str) -> None:
        self.nex = nexdata
        self.filename = filename
        self.animal, self.date, _ = parse_filename(filename)
        self.ensemble = False
        self.params = Params()
        self.lick = 'Lick'
        self.rinse = 'Rinse'
        self.all_events = (self.params.tastants_opts + (self.lick, self.rinse))
        self.timestamps = {}
        self.reinforced_licks = []
        self.drylicks = []
        self.neurons = ()
        self.trial_times = {}
        self.drylick_trials = []
        self.tastants = self.params.tastants_opts
        self.drylick = 'drylick'
        self.final_df = {}
        self.__populate()
        self.all_responses = {neuron: {'LXL': False, 'COH': 'not run', '5L': False} for neuron in self.neurons}

    def __repr__(self) -> str:
        ens = '-E' if self.ensemble else ''
        return f'{self.__class__.__name__}{ens}'

    @property
    def fl_trials(self):
        try:
            trials = self.final_df['5L'].pivot(index='Neuron', columns='Stimulus', values='Trial Count')
            trials.insert(0, 'trials', ['trials' for x in range(trials.shape[0])])
            return trials
        except KeyError:
            return None

    @property
    def fl_adjmags(self):
        try:
            mag = self.final_df['5L'].pivot(index='Neuron', columns='Stimulus', values='Adj. Magnitude')
            mag.insert(0, 'adj.magnitude', ['adj.magnitude' for x in range(mag.shape[0])])
            return mag
        except KeyError:
            return None

    @property
    def fl_baseline(self):
        try:
            baseline = self.final_df['5L'].pivot(index='Neuron', columns='Stimulus', values='Baseline')
            baseline.insert(0, 'baseline', ['baseline' for x in range(baseline.shape[0])])
            return baseline
        except KeyError:
            return None

    @property
    def fl_latency(self):
        try:
            latency = self.final_df['5L'].pivot(index='Neuron', columns='Stimulus', values='Latency')
            latency.insert(0, 'latency', ['latency' for x in range(latency.shape[0])])
            return latency
        except KeyError:
            return None

    @property
    def fl_duration(self):
        try:
            duration = self.final_df['5L'].pivot(index='Neuron', columns='Stimulus', values='Duration')
            duration.insert(0, 'duration', ['duration' for x in range(duration.shape[0])])
            return duration
        except KeyError:
            return None

    @property
    def fl_respstat(self):
        respstat = self.calculate_statistics()
        return respstat

    @property
    def fl_respstat_mag(self):
        respstat = self.fl_respstat
        return respstat.pivot(index='Neuron', columns='Tastant', values='Magnitude')

    @property
    def fl_respstat_lat(self):
        respstat = self.fl_respstat
        return respstat.pivot(index='Neuron', columns='Tastant', values='Latency')

    @property
    def fl_respstat_dur(self):
        respstat = self.fl_respstat
        return respstat.pivot(index='Neuron', columns='Tastant', values='Duration')

    @property
    def fl_respstat_sig(self):
        respstat = self.fl_respstat
        return respstat[respstat['Magnitude'].notnull()]

    @property
    def lxl_responses(self):
        try:
            lxl = self.final_df['lxl'].pivot(index='Neuron', columns='Tastant', values='Response Magnitude')
            lxl.insert(0, 'LxL', ['lxl' for x in range(lxl.shape[0])])
            return lxl
        except KeyError:
            return None

    @property
    def lxl_trials(self):
        try:
            trials = self.final_df['lxl'].pivot(index='Neuron', columns='Tastant', values='Trial Count')
            trials.insert(0, 'trials', ['trials' for x in range(trials.shape[0])])
            return trials
        except KeyError:
            return None

    @staticmethod
    def get_spike_times_within_interval(timestamps, start, end):
        return timestamps[(timestamps >= start) & (timestamps <= end)]

    def reorder_stats_columns(self, df):
        existing_cols = [col for col in self.params.order if col in df.columns]
        remaining_cols = [col for col in df.columns if col not in existing_cols]
        new_cols = existing_cols + remaining_cols
        return df[new_cols]

    def __populate(self):
        count = 0
        self.timestamps['Start'] = np.asarray([0.0])
        self.timestamps['Stop'] = np.asarray([0.0])

        for var in self.nex['Variables']:
            if (var['Header']['Name'] in self.all_events) or (var['Header']['Type'] == 0):
                self.timestamps[var['Header']['Name']] = var['Timestamps']
            if var['Header']['Type'] == 0:
                self.neurons += (var['Header']['Name'],)
                count += 1

        self.ensemble = count > 1
        for key, value in self.timestamps.items():
            if max(value) > self.timestamps['Stop'][0]:
                self.timestamps['Stop'] = np.asarray([max(value)])

            if not key == self.lick and key in self.all_events:
                corrected_values = []
                for ts in value:
                    correct_time = \
                        np.where((self.timestamps[self.lick] < ts + .005) & (self.timestamps[self.lick] > ts - .005))[0]
                    if len(correct_time) == 1:
                        corrected_values.append(self.timestamps[self.lick][correct_time][0])
                    elif len(correct_time) == 0:
                        print('Lick removed from {} due to not being in lick variable'.format(key))
                    else:
                        sys.exit("Not the correct amount of licks")

                self.timestamps[key] = np.array(corrected_values)
                self.reinforced_licks.extend(corrected_values)

        self.drylicks = [lick_event for lick_event in self.timestamps['Lick'] if
                         lick_event not in self.reinforced_licks]
        self.timestamps[self.drylick] = np.array(self.drylicks)

        for tastant in self.tastants:
            tastant1 = []
            for ts in self.timestamps[tastant]:
                if self.timestamps[self.lick][np.where(self.timestamps[self.lick] == ts)[0] - 1] in self.timestamps[
                    self.drylick]:
                    tastant1.append(ts)
                    self.drylick_trials.append(
                        self.timestamps[self.lick][np.where(self.timestamps[self.lick] == ts)[0] - 1][0])
            self.trial_times[tastant] = np.array(tastant1)
        # remove tastants with too many trials
        self.trial_times = {trial: arr for trial, arr in self.trial_times.items() if arr.size <= 25}
        self.tastants = [tastant for tastant in self.tastants if tastant in self.trial_times.keys()]
        self.trial_times[self.drylick + '_lxl'] = np.array(self.drylick_trials)

    def run_stats(self, functions_to_run):
        """ Runs the functions specified in the functions_to_run list.
        If 'all' is in the list, all functions will be run.
        Options:
            'coh': self.coherence,
            'ili': self.ili,
            'spont': self.spont,
            'baseline': self.baseline,
            'changepoint': self.changepoint,
            'chisquareLXL': self.chisquareLXL
            'respstat': self.respstat

        """
        function_map = {
            'coh': self.coherence,
            'ili': self.ili,
            'spont': self.spont,
            'baseline': self.baseline,
            'changepoint': self.changepoint,
            'chisquare': self.chisquareLXL,
            'respstat': self.calculate_statistics
        }

        for function_name in functions_to_run:
            if 'all' in function_name:
                for func in function_map:
                    function_map[func]()
            elif function_name in function_map:
                function_map[function_name]()
            else:
                print(f"Function '{function_name}' not found.")

    def spont(self, spontbin=1):
        all_spont = []
        spont_times = self.spont_intervals(
            self.timestamps[self.lick], self.timestamps['Start'], self.timestamps['Stop'], 3, 1
        )

        for neuron in self.neurons:
            if len(spont_times) > 0:
                spontbins = []

                for start, end in spont_times:
                    mask = (self.timestamps[neuron] >= start) & (self.timestamps[neuron] < end)
                    spont_spikes = self.timestamps[neuron][mask] - start

                    bins = range(0, math.floor((end - start)) * spontbin + 1, spontbin)
                    hist, _ = np.histogram(spont_spikes, bins)
                    spontbins.extend((hist / spontbin))

                spontbins = np.array(spontbins)
                n = spontbins.size

                mean_spont = spontbins.mean()
                std_spont = spontbins.std(ddof=1)
                sem_spont = std_spont / np.sqrt(n)

                all_spont.append({
                    'File Name': self.filename,
                    'Neuron': neuron,
                    'Mean Spont': mean_spont,
                    'SEM Spont': sem_spont,
                    'STD Spont': std_spont,
                })

        self.final_df['spont'] = pd.DataFrame(all_spont)

    def baseline(self):
        all_baseline = []
        for neuron in self.neurons:
            baselines = []
            for key, value in self.trial_times.items():  # for each tastant
                if 'lxl' in key:  # skip if drylick
                    continue
                allspikes = []
                trial_count = len(value)
                for trial_time in value:  # for each trial, get spikes centered around trial time
                    allspikes.extend(
                        (self.timestamps[neuron][np.where((self.timestamps[neuron] > trial_time - self.params.base) & (
                                self.timestamps[neuron] < trial_time + self.params.trial_end))] - trial_time).tolist())
                allbins = np.histogram(allspikes, np.linspace(-self.params.base, self.params.trial_end, num=int(
                    (self.params.trial_end + self.params.base) / self.params.bin5L + 1)))[
                    0]  # create histogram of spikes
                baselines.append(np.mean(allbins[0:20]) / (self.params.bin5L * trial_count))
            all_baseline.append({'File Name': self.filename, 'Neuron': neuron, 'Baseline Mean': np.mean(
                baselines), 'Baseline STDEV': np.std(baselines, ddof=1)})
        self.final_df['baseline'] = pd.DataFrame(all_baseline)

    def chisquareLXL(self):
        all_LXL = []
        bin_edges = np.arange(0, 0.150 + 0.015, 0.015)

        for neuron in self.neurons:
            LXL_data = {}
            dryspikes = []
            # get all spike timestamps around drylick activity, time adjusted around each drylick
            for i in range(len(self.trial_times[self.drylick + '_lxl'])):
                dryspikes.extend(
                    self.timestamps[neuron][
                        np.where((self.timestamps[neuron] >= self.trial_times[self.drylick + '_lxl'][i]) & (
                                self.timestamps[neuron] < (
                                    self.trial_times[self.drylick + '_lxl'][i] + self.params.lick_max)))] -
                    self.trial_times[self.drylick + '_lxl'][i])
            # number of drylick trial-related neuron spikes
            ndry = len(dryspikes)
            dryhist = np.histogram(dryspikes, np.arange(
                0, self.params.lick_max + self.params.lick_window, self.params.lick_window))[0]  # bin the spikes
            for tastant in self.tastants:  # run chi square
                lxl_datum = self.LXL(neuron, tastant, self.timestamps[tastant], self.timestamps[neuron], dryhist, ndry,
                                     lick_max=self.params.lick_max,
                                     lick_window=self.params.lick_window, wil_alpha=self.params.wil_alpha,
                                     chi_alpha=self.params.chi_alpha)
                all_LXL.append(lxl_datum)
        self.final_df['lxl'] = pd.DataFrame(all_LXL)
        xxx = 5

    def changepoint(self):
        def process_changepoints(changepoints):
            change_times = (np.asarray(changepoints[:-1]) - 20) / 10
            close_to_zero = np.where((change_times < 0) & (change_times >= -0.3))[0]
            not_too_close = np.where((change_times >= 0) & (change_times < 0.3))[0]

            if close_to_zero.size > 0 and not_too_close.size == 0:
                changepoints[close_to_zero] = 20
                change_times[close_to_zero] = 0

            valid_indices = np.where(change_times >= 0)
            return change_times[valid_indices], changepoints[valid_indices]

        def get_response_stats(neuron, allbins, changepoints, change_times):
            if len(change_times) == 1 and change_times[0] > 3.5 or len(change_times) == 0:
                return 'None', 'None', 'None', 'None'

            if len(change_times) == 1:
                self.all_responses[neuron]['5L'] = True
                respbins = allbins[changepoints[0]:]
                duration = 4 - change_times[0]
            else:
                self.all_responses[neuron]['5L'] = True
                respbins = allbins[changepoints[0]:changepoints[1]]
                duration = change_times[1] - change_times[0]

            magnitude = np.mean(respbins) / (0.1 * trial_count)
            adj_magnitude = magnitude - baseline
            latency = change_times[0]

            return magnitude, adj_magnitude, latency, duration

        columns = ['File', 'Neuron', 'Stimulus', 'Trial Count', 'Baseline', 'Magnitude',
                   'Adj. Magnitude', 'Latency', 'Duration', 'CP1', 'CP2', 'CP3', 'CP4', 'Init CPA',
                   'eff. alpha', 'pen']
        file_5L = pd.DataFrame(columns=columns)

        for neuron in self.neurons:
            bootpen, eff_a = self.get_5L_penalty(self.timestamps[neuron], 1000, 0.05, 0.005)
            print(f'Effective 5-Lick alpha for {neuron} is {eff_a}.')
            trial_times = {
                key: value for key, value in self.trial_times.items() if 'lxl' not in key
            }

            for key, value in trial_times.items():
                trial_count = len(value)
                allspikes = [spike - trial_time for trial_time in value for spike in self.timestamps[neuron]
                             if trial_time - 2 < spike < trial_time + 4]
                allbins = np.histogram(allspikes, np.linspace(-2, 4, num=int((4 + 2) / 0.1 + 1)))[0]

                algo = rpt.Pelt(model='l1', min_size=3, jump=1).fit(allbins)
                changepoints = np.array(algo.predict(pen=bootpen))
                change_times, changepoints = process_changepoints(changepoints)
                init = np.copy(change_times)
                baseline = np.mean(allbins[0:20]) / (0.1 * trial_count)
                magnitude, adj_magnitude, latency, duration = get_response_stats(neuron, allbins, changepoints,
                                                                                 change_times)
                this_5L = {'File': self.filename,
                           'Neuron': neuron,
                           'Stimulus': key,
                           'Trial Count': trial_count,
                           'Baseline': baseline,
                           'Magnitude': magnitude,
                           'Adj. Magnitude': adj_magnitude,
                           'Latency': latency,
                           'Duration': duration,
                           'CP1': 'nan',
                           'CP2': 'nan',
                           'CP3': 'nan',
                           'CP4': 'nan',
                           'Init CPA': init,
                           'eff alpha': eff_a,
                           'pen': bootpen
                           }
                for n in range(len(change_times)):
                    if len(change_times) <= 4:
                        this_5L['CP' + str(n + 1)] = change_times[n]
                    else:
                        this_5L['CP' + str(n + 1)] = f'{len(change_times)} change points found'
                file_5L = file_5L.append(this_5L, ignore_index=True)
        self.final_df['5L'] = file_5L

    def coherence(self):
        binsize, numbin = 1 / (2 * 50), 2 * 256
        coh_data = pd.DataFrame(columns=['File', 'Neuron', 'coh peak (4-9Hz)', 'coh peak (6-8Hz)',
                                         'coh peak (10-15Hz)', 'coh mean (4-9)', 'coh mean (6-8)', 'coh mean (10-15)'])
        # bin size and number of bins per segment
        lick_bin = np.histogram(self.timestamps[self.lick], np.arange(
            self.timestamps['Start'], self.timestamps['Stop'], binsize))[0]  # binning licks
        # see Kattla S, Lowery MM. for the following section
        L = len(lick_bin) / numbin
        Lstar = math.floor((L - 1) / (1 - .5)) + 1
        win = signal.get_window('hann', numbin)
        wbot = sum([el ** 2 for el in win])
        wtop = sum([win[i] * win[i + int((1 - .5) * numbin)] for i in range(0, int(256 * 2 * .5))])
        wprime = (wtop / wbot) ** 2
        w = 1 / (1 + 2 * wprime)
        # condifence cuttof based on user specified alpha value , see Kattla
        Z = 1 - .01 ** (1 / (w * Lstar - 1))
        # end Kattle section
        for neuron in self.neurons:
            # bin the spikes
            cell_bin = np.histogram(self.timestamps[neuron],
                                    np.arange(self.timestamps['Start'], self.timestamps['Stop'], binsize))[0]

            f, Cxy = signal.coherence(lick_bin, cell_bin, fs=1 / binsize, nperseg=numbin,
                                      window='hann', detrend=False, noverlap=numbin / 2)
            # get coherence ranges
            four, six, ten = Cxy[np.where((f > 4) & (f < 9))], Cxy[np.where(
                (f > 6) & (f < 8))], Cxy[np.where((f > 10) & (f < 15))]
            coherent = False
            # if the user specified number of consecutive bins are significant
            for i in range(0, len(four) - 3):
                if np.all(four[i:i + 3] > Z):
                    coherent = True
            if coherent:  # record coherence data
                self.all_responses[neuron]['COH'] = True
                coh_data.loc[len(coh_data)] = [self.filename, neuron, max(four), max(
                    six), max(ten), np.mean(four), np.mean(six), np.mean(ten)]
            else:  # if not coherent, nan
                self.all_responses[neuron]['COH'] = False
                coh_data = coh_data.append(
                    [pd.Series(dtype='float64')], ignore_index=True)
                coh_data.loc[len(coh_data) - 1, 0:2] = [self.filename, neuron]
        self.final_df['COH'] = pd.DataFrame(coh_data)

    def ili(self):
        all_ILIs = []
        ILIs = {}
        for tastant in self.tastants:  # if all trials are 5 licks, this will be skipped
            while len(self.timestamps[tastant]) / 5 > len(self.trial_times[tastant]):
                # remove any licks that followed another lick too quickly (must be artifact)
                self.timestamps[tastant] = np.delete(self.timestamps[tastant],
                                                     np.where(np.diff(self.timestamps[tastant]) <= 0.1)[0] + 1)
                for i in range(len(self.trial_times[tastant]) - 1):  # for each trial
                    # if this trial had more than 5 licks
                    if len(np.where((self.timestamps[tastant] >= self.trial_times[tastant][i]) & (
                            self.timestamps[tastant] < self.trial_times[tastant][i + 1]))[0]) > 5:
                        self.timestamps[tastant] = np.delete(self.timestamps[tastant],
                                                             np.where((self.timestamps[tastant] >=
                                                                       self.trial_times[tastant][i]) & (
                                                                              self.timestamps[tastant] <
                                                                              self.trial_times[tastant][i + 1]))[0][
                                                                 -1])  # delete the final lick of the trial
                    # if the last trial had more than 5 licks
                    if len(np.where(self.timestamps[tastant] >= self.trial_times[tastant][-1])[0]) > 5:
                        self.timestamps[tastant] = np.delete(self.timestamps[tastant], len(
                            self.timestamps[tastant]) - 1)  # remove the final lick
                        break
            ILItimes = np.sort(self.timestamps[tastant])  # get tastant timestamps
            tasteILI = np.diff(ILItimes)  # get tastant ILIs
            ILIbreaks = []
            time5L = []

            for i in range(1, len(ILItimes)):
                if ILItimes[i] in self.trial_times[tastant]:
                    ILIbreaks.append(i - 1)
            # adds the beginning and the end so that we get all the trial completion times
            break5L = [-1] + ILIbreaks + [len(tasteILI)]
            # loops through and calculates trial completion times
            for i in range(len(break5L) - 1):
                if len(tasteILI[break5L[i] + 1:break5L[i + 1]]) == 4:
                    time5L.append(sum(tasteILI[break5L[i] + 1:break5L[i + 1]]))
            # remove inter-trial intervals from the ILIs
            tasteILI = np.delete(tasteILI, ILIbreaks)
            ILIs[tastant] = {'intervals': tasteILI, 'trialcount': len(
                self.trial_times[tastant]), '5L durations': time5L}  # package ILI info
        all_ILIs.append(ILIs)

        allILIdata = pd.DataFrame(index=self.tastants,
                                  columns=['Median ILI', 'Pause Count', 'Mean Pause Length', 'Mean Pause Length SEM',
                                           'Median Pause Length', 'Mean Time to Complete 5L',
                                           'Mean Time to Complete 5L SEM', 'Median Time to Complete 5L', 'Trial Count'])
        allILIs = {}
        for tastant in self.tastants:
            tastant_ILIs = []
            tastant_5L_times = []
            trial_count = 0
            for fileili in all_ILIs:  # compile ILI info across all sessions
                tastant_ILIs.extend(fileili[tastant]['intervals'])
                tastant_5L_times.extend(fileili[tastant]['5L durations'])
                trial_count += fileili[tastant]['trialcount']
            # turn into array for numpy indexing
            ilifinal = np.asarray(tastant_ILIs)
            fivefinal = np.asarray(tastant_5L_times)
            # calculate stats for each tastant, and store in the dataframe
            allILIdata.loc[tastant] = [np.median(ilifinal), len(np.where(ilifinal >= 1)[0]),
                                       np.mean(ilifinal[np.where(ilifinal >= 1)[0]]), stats.sem(ilifinal[np.where(
                    ilifinal >= 1)[0]]), np.median(ilifinal[np.where(ilifinal >= 1)[0]]), np.mean(fivefinal),
                                       stats.sem(fivefinal), np.median(fivefinal), trial_count]
            allILIs[tastant] = ilifinal
        self.final_df['ILI'] = pd.DataFrame.from_dict(
            {key: pd.Series(value) for key, value in allILIdata.items()})

    def calculate_trial_response(self, binned_spike_times, bins, trial, neuron, tastant, mean_spont, std_spont):
        window_size = 3
        is_counting = False
        responses = []
        start_index = None
        for i in range(len(binned_spike_times) - window_size + 1):
            window = binned_spike_times[i: i + window_size]
            if window.sum() > (mean_spont + std_spont):
                if not is_counting:
                    start_index = i
                    is_counting = True
            else:
                if is_counting:  # end of a response
                    end_index = i - 1
                    responses.append(self.calculate_response(bins, start_index, end_index, trial, neuron, tastant,
                                                             mean_spont, std_spont, binned_spike_times))
                    is_counting = False

        # check if the counting ended with the last window
        if is_counting:
            end_index = len(binned_spike_times) - window_size
            responses.append(self.calculate_response(bins, start_index, end_index, trial, neuron, tastant, mean_spont,
                                                     std_spont, binned_spike_times))

        if not responses:
            return [[neuron, tastant, trial, mean_spont, std_spont, np.nan, np.nan, np.nan]]

        return responses

    @staticmethod
    def calculate_response(bins, start_index, end_index, trial, neuron, tastant, mean_spont, std_spont,
                           binned_spike_times):
        latency = bins[start_index] - trial  # calculate latency
        duration = bins[end_index] - bins[start_index]
        # calculate magnitude by summing window and dividing by duration
        magnitude = binned_spike_times[start_index: end_index + 1].sum() / duration
        resp = [neuron, tastant, trial, mean_spont, std_spont, magnitude, latency, duration]
        return resp

    def calculate_statistics(self, binsize=1):
        """Calculate response statistics for each neuron."""

        spont_times = self.spont_intervals(
            self.timestamps[self.lick], self.timestamps['Start'], self.timestamps['Stop'], 3, 1
        )
        allresp = []
        for neuron in self.neurons:
            # spont section (std not sem)
            spontbins = []
            for start, end in spont_times:
                spont_spikes = self.timestamps[neuron][
                                   (self.timestamps[neuron] >= start) &
                                   (self.timestamps[neuron] < end)
                                   ] - start
                bins = range(0, math.floor((end - start) / 1) * binsize + 1, binsize)
                hist, _ = np.histogram(spont_spikes, bins)
                spontbins.extend((hist / binsize).tolist())

            mean_spont = np.mean(spontbins)
            std_spont = np.std(spontbins)
            sem_spont = stats.sem(spontbins)

            for tastant in self.tastants:
                trials = self.trial_times[tastant]
                for trial in trials:

                    spks = self.get_spike_times_within_interval(self.timestamps[neuron], trial, trial + 4)
                    bins = np.arange(spks.min(), spks.max() + 0.1, 0.1)
                    binned_spike_times, _ = np.histogram(spks, bins=bins)

                    trial_response = self.calculate_trial_response(binned_spike_times, bins, trial, neuron, tastant,
                                                                   mean_spont, std_spont)
                    if not trial_response:
                        trial_response = [neuron, tastant, trial, mean_spont, std_spont, np.nan, np.nan, np.nan]
                    allresp.extend(trial_response)

        return pd.DataFrame(allresp, columns=['Neuron', 'Tastant', 'Trial', 'Spont_m', 'Spont_std', 'Magnitude',
                                              'Latency', 'Duration'])

    def temp_calculate_statistics(self, binsize=1):
        """Calculate response statistics for each neuron."""

        spont_times = self.spont_intervals(
            self.timestamps[self.lick], self.timestamps['Start'], self.timestamps['Stop'], 3, 1
        )
        allresp = pd.DataFrame(
            columns=['Animal', 'Neuron', 'Tastant', 'Trial', 'Spont_m', 'Spont_std', 'Magnitude', 'Latency',
                     'Duration'])
        for neuron in self.neurons:
            # spont section (std not sem)
            spontbins = []
            for start, end in spont_times:
                spont_spikes = self.timestamps[neuron][
                                   (self.timestamps[neuron] >= start) &
                                   (self.timestamps[neuron] < end)
                                   ] - start
                bins = range(0, math.floor((end - start) / 1) * binsize + 1, binsize)
                hist, _ = np.histogram(spont_spikes, bins)
                spontbins.extend((hist / binsize).tolist())

            mean_spont = np.mean(spontbins)
            std_spont = np.std(spontbins)
            sem_spont = stats.sem(spontbins)

            for tastant in self.tastants:
                tastant_response = pd.DataFrame(columns=allresp.columns)
                trials = self.trial_times[tastant]
                for trial in trials:
                    trial_response = pd.DataFrame(columns=allresp.columns)
                    spks = self.get_spike_times_within_interval(self.timestamps[neuron], trial, trial + 4)
                    bins = np.arange(spks.min(), spks.max() + 0.1, 0.1)
                    binned_spike_times, _ = np.histogram(spks, bins=bins)

                    window_size = 3
                    is_counting = False
                    start_index = 0

                    for i in range(len(binned_spike_times) - window_size + 1):
                        window = binned_spike_times[i: i + window_size]
                        if window.sum() > (mean_spont + std_spont):
                            if not is_counting:
                                start_index = i
                                is_counting = True
                        else:
                            if is_counting:  # end of a response
                                end_index = i - 1
                                latency = bins[start_index] - trial  # calculate latency
                                duration = bins[end_index] - bins[start_index]
                                # calculate magnitude by summing window and dividing by duration
                                magnitude = binned_spike_times[start_index: end_index + 1].sum() / duration
                                resp = [self.animal, neuron, tastant, trial, mean_spont, std_spont, magnitude, latency,
                                        duration]
                                resp_df = pd.DataFrame([resp], columns=allresp.columns)
                                trial_response = trial_response.append(resp_df, ignore_index=True)
                                is_counting = False

                    # check if the counting ended with the last window
                    if is_counting:
                        end_index = len(binned_spike_times) - window_size
                        latency = bins[start_index] - trial
                        duration = bins[end_index] - bins[start_index]
                        magnitude = binned_spike_times[start_index: end_index + 1].sum() / duration
                        resp = [self.animal, neuron, tastant, trial, mean_spont, std_spont, magnitude, latency,
                                duration]
                        resp_df = pd.DataFrame([resp], columns=allresp.columns)
                        trial_response = trial_response.append(resp_df, ignore_index=True)

                    if trial_response.empty:
                        resp = [self.animal, neuron, tastant, trial, mean_spont, std_spont, np.nan, np.nan,
                                np.nan]
                        resp_df = pd.DataFrame([resp], columns=allresp.columns)
                        trial_response = trial_response.append(resp_df, ignore_index=True)
                tastant_response = tastant_response.append(trial_response, ignore_index=True)
            allresp = allresp.append(tastant_response, ignore_index=True)
        return allresp

    def get_5L_penalty(self, neur, bootstrap_n=1000, boot_alpha=.05, alpha_tol=.005):
        """Obtain penalty value for us in change-point calculation."""
        base = 2  # time before trial start to include in histogram
        trial_end = 5  # time after trial start to include in histogram
        bin5L = 0.1  # bin size for histogram
        model = 'l1'  # model for change-point calculation
        bootbins = []
        for key, value in self.trial_times.items():  # for each tastant
            if 'lxl' in key:  # skip if drylick
                continue
            allspikes = []
            # for each trial, collect the spikes, normalized for trial time
            for trial_time in value:
                allspikes.extend((neur[np.where(
                    (neur > trial_time - base) & (neur < trial_time + trial_end))] - trial_time).tolist())
            # create histogram of spikes
            allbins = np.histogram(allspikes, np.linspace(-base, trial_end, num=int((trial_end + base) / bin5L + 1)))[0]
            bootbins.extend(allbins)  # add the bins to the list for bootstrapping
        random.seed(a=34)  # set random seed
        bootpen = 1
        pelts = []
        fp = np.ones(bootstrap_n)
        scale = .5
        old_a, old_bootpen, eff_a = 1, bootpen, 1
        # Get bootstrap_n number of fitted models
        # for a random sample of bins from the data
        for i in range(bootstrap_n):
            pelts.append(rpt.Pelt(model=model, min_size=3, jump=1).fit(
                np.array(random.sample(bootbins, k=len(allbins)))))
        # loop will continue if the effective alpha has not reached desired alpha
        while eff_a > boot_alpha:
            old_fp = np.copy(fp)  # save old run results
            old_a = eff_a  # save olf run alpha
            # for each fitted model, get changepoints
            for ind in np.where(fp != 0)[0]:
                fp[ind] = len(pelts[ind].predict(bootpen)) - 1
            # calculate the effective alpha
            # (number of false positives over number of bootstraps)
            eff_a = np.count_nonzero(fp) / bootstrap_n
            if eff_a < boot_alpha:  # if the effective alpha overshot
                if scale < .001 or boot_alpha - eff_a < alpha_tol:
                    # if the scale has gotten too small or
                    # eff_a is within tolerance, exit
                    break
                # reduce scale (makes adjustments to penalty more sensitive)
                scale = scale / 4
                bootpen = old_bootpen  # reset
                fp = old_fp
                eff_a = old_a
            else:
                # save old bootpen in case the new one overshoots
                old_bootpen = bootpen
                # increase the penalty value
                bootpen += (eff_a - boot_alpha) * 100 * scale / eff_a
        return bootpen, eff_a

    def LXL(self, neuron, tastant, stim, neur, dryhist, ndry, lick_max=.150, lick_window=.015, wil_alpha=.05,
            chi_alpha=.05):
        """Run a lick-by-lick taste response calculation."""
        tastespikes = []
        # get all spike timestamps around tastant activity,
        # time adjusted around each tastant lick
        for i in range(len(stim)):
            tastespikes.extend(neur[np.where((neur >= stim[i]) & (neur < (stim[i] + lick_max)))] - stim[i])
        ntaste = len(tastespikes)  # number of taste spikes
        tasteobs = np.histogram(tastespikes, np.arange(0, lick_max + lick_window, lick_window))[
            0]  # bin the tastant spikes
        expected = dryhist * ntaste / ndry
        if sum(expected) >= 10:
            expected, tasteobs = self.binmerge(expected, tasteobs)
            chi_stat, gof_p = stats.chisquare(tasteobs, f_exp=expected)  # run chi square
            # get firing rates for drylick and tastant
            dryfire = (expected * ndry / ntaste) / len(self.trial_times[self.drylick + '_lxl']) / lick_window
            tastefire = tasteobs / len(stim) / lick_window
            respmag = np.mean(np.absolute(tastefire - dryfire))  # response magnitude
            resptype = 0
            wstat, pval = stats.wilcoxon(dryfire, tastefire)  # run wilcoxon
            if pval < wil_alpha:  # if wilcoxon is significant, add response type
                if sum(tastefire - dryfire) > 0:
                    resptype = 'Excitatory'
                elif sum(tastefire - dryfire) < 0:
                    resptype, respmag = 'Inhibitory', respmag * -1
            else:
                resptype = 'Time Course'
        else:
            gof_p = 1
        q_value = gof_p * len(self.tastants)  # bonferroni correction
        # if the chi square is significant, record the taste response
        if q_value < chi_alpha:
            self.all_responses[neuron]['LXL'] = True
            this_LXL = {'File Name': self.filename,
                        'Neuron': neuron,
                        'Tastant': tastant,
                        'Trial Count': len(stim),
                        'P-Value': gof_p,
                        'Q-Value': q_value,
                        'Response Magnitude': respmag,
                        'Wilcoxon_p': pval,
                        'Response_Type': resptype}
        else:
            self.all_responses[neuron]['LXL'] = False
            this_LXL = {'File Name': self.filename,
                        'Neuron': neuron,
                        'Tastant': tastant,
                        'Trial Count': len(stim),
                        'P-Value': gof_p,
                        'Q-Value': q_value,
                        'Response Magnitude': 'nan',
                        'Wilcoxon_p': 'nan',
                        'Response_Type': 'nan'}
        return this_LXL

    @staticmethod
    def fdr(pvals, q, stimuli):
        """computes false discovery adjusted p-values for multiple comparisons.
        Adapted from code written by Jonathan Viktor."""
        fdr_data = pd.DataFrame(list(zip(stimuli, pvals)), columns=['stim', 'p'])
        fdr_data = fdr_data.sort_values('p')
        p = list(fdr_data['p'])
        V = len(p)
        I = np.arange(1, V + 1)
        cVN = sum(1 / I)
        pN = (I / V) * (q / cVN)
        if not np.any(p < pN):
            sigs = ["nonsig" for i in range(V)]
        else:
            lastsig = max(np.where(p < pN)[0])
            sigs = ["sig" if i <= lastsig else "nonsig" for i in range(V)]
        fdr_data.insert(2, 'sigs', sigs)
        fdr_data.insert(2, 'pN', pN)
        return fdr_data

    @staticmethod
    def binmerge(expected, observed):
        """ Merges bins in the case where bin counts would invalidate chi square. """
        e = np.array(expected)
        o = np.array(observed)
        if len(np.where(e < 1)[0]) > 0:  # if any bins are less than one
            for this_bin in np.where(e < 1)[0]:  # merge each bin with the following bin
                if this_bin == len(e) - 1: this_bin = this_bin - 1  # if the last bin move counter back one
                e[this_bin + 1], o[this_bin + 1] = e[this_bin + 1] + e[this_bin], o[this_bin + 1] + o[
                    this_bin]  # merge bins
                e[this_bin], o[this_bin] = 0, 0
            o = o[np.where(e != 0)[0]]  # remove 0 bins
            e = e[np.where(e != 0)[0]]
        while len(np.where(e < 5)[0]) / len(e) > .2 and len(e) > 1:  # greater than 80% of bins must be 5 or higher
            this_bin = np.where(e < 5)[0][0]  # find first offending bin and merge
            if this_bin == len(e) - 1: this_bin = this_bin - 1
            e[this_bin + 1], o[this_bin + 1] = e[this_bin + 1] + e[this_bin], o[this_bin + 1] + o[this_bin]
            e[this_bin], o[this_bin] = 0, 0
            o = o[np.where(e != 0)[0]]
            e = e[np.where(e != 0)[0]]
        return e, o

    @staticmethod
    def spont_intervals(licks, start, stop, prespont, postspont):
        """Determine the intervals for spontaneous activity evaluation."""
        # find ILIs long enough to be a spontaneous interval
        spont_indices = np.where(np.diff(licks) >= 10)[0]
        # spontaneous interval start and end times
        spont_starts = licks[spont_indices] + prespont
        spont_ends = licks[spont_indices + 1] - postspont  # end times
        # get the interval for the beginning, if there is one
        if licks[0] - start[0] >= 10:
            spont_starts = np.sort(np.append(spont_starts, 0))
            spont_ends = np.sort(np.append(spont_ends, licks[0] - postspont))
        # get the interval for the end, if there is one
        if stop[0] - licks[-1] >= 10:
            spont_starts = np.sort(np.append(spont_starts, licks[-1] + prespont))
            spont_ends = np.sort(np.append(spont_ends, stop[0]))
        # combine the start and end times into a tuple of tuples
        spont_times = tuple(zip(spont_starts, spont_ends))
        return spont_times
