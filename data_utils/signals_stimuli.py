from __future__ import annotations
import warnings
import sys
from collections import namedtuple
import ruptures as rpt
import numpy as np
import scipy.stats as stats
from scipy import signal
import pandas as pd
import math
import random
from params import Params
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

class StimuliSignals:
    def __init__(self, nexdata: dict, filename: str) -> None:
        self.filename = filename
        self.nex = nexdata
        self.ensemble = False
        self.params = Params()
        self.lick = 'Lick'
        self.rinse = 'Rinse'
        self.all_events = (self.params.tastants + (self.lick, self.rinse))


        self.timestamps = {}
        self.reinforced_licks = []
        self.drylicks = []
        self.neurons = ()
        self.trial_times = {}
        self.drylick_trials = []
        self.tastants = self.params.tastants
        self.drylick = 'drylick'
        self.all_LXL, self.all_spont, self.all_COH, self.all_5L, self.all_ILI, self.all_baseline = [], [], [], [], [], []
        self.final_df = {}
        self.__populate()

    def __repr__(self) -> str:
        ens = '-E' if self.ensemble else ''
        return f'{self.__class__.__name__}{ens}'

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
                self.neurons = self.neurons+tuple([var['Header']['Name']])
                count += 1
        self.ensemble = True if count > 1 else False

        for key, value in self.timestamps.items():
            if max(value) > self.timestamps['Stop'][0]:
                self.timestamps['Stop'] = np.asarray([max(value)])
            # if it is not licks, and it is not a neuron
            if not key == self.lick and key in self.all_events:
                for ts in value:  # for each timestamp
                    # if the timestamp is not in the lick array
                    if ts not in self.timestamps[self.lick]:
                        # find the associated timestamp in the lick array
                        correct_time = \
                        np.where((self.timestamps[self.lick] < ts + .005) & (self.timestamps[self.lick] > ts - .005))[0]
                        # there should only be one timestamp near the incorrect one
                        if len(correct_time) == 1:
                            # if so, replace the original with
                            # the correct timestamp
                            value[np.where(value == ts)[0]
                            ] = self.timestamps[self.lick][correct_time]
                        elif len(correct_time) == 0:
                            value = np.delete(value, np.where(value == ts)[0])
                            print('Lick removed from {} due to not being in lick variable'.format(key))
                        else:
                            sys.exit("Not the correct amount of licks")
                # this just checks through to make
                # sure all the licks were corrected
                for ts in value:
                    if ts not in self.timestamps[self.lick]:
                        sys.exit("man we missed one of the licks")
                    else:
                        # add this to the list of reinforced licks
                        self.reinforced_licks.append(ts)
                # replace the old values with the corrected values
                self.timestamps[key] = value
        for lick_event in self.timestamps['Lick']:  # get drylicks
            if lick_event not in  self.reinforced_licks:  # if it's not a reinforced lick
                self.drylicks.append(lick_event)  # it's a drylick
        # add the drylicks to the timestamps dict
        self.timestamps[self.drylick] = np.array(self.drylicks)
        for tastant in self.tastants:  # get starts of 5-lick trials
            tastant1 = []
            # go through each timestamp, and
            # determine whether the previous lick was a drylick
            for ts in self.timestamps[tastant]:
                if self.timestamps[self.lick][np.where(self.timestamps[self.lick] == ts)[0]-1] in self.timestamps[self.drylick]:
                    # if so , that was the first lick of the trial
                    tastant1.append(ts)
                    self.drylick_trials.append(self.timestamps[self.lick][np.where(self.timestamps[self.lick] == ts)[0]-1][0])
            # trial start times used for 5L and ILI
            self.trial_times[tastant] = np.array(tastant1)
        self.trial_times[self.drylick+'_lxl'] = np.array(self.drylick_trials)

    def get_allstats(self):
        self.coherence()
        self.coherence()
        self.ili()
        self.spont()
        self.baseline()
        self.changepoint()
        self.chisquareLXL()

    def spont(self, spontbin=1):
        spontbins = None
        spont_times = self.spont_intervals(
            self.timestamps[self.lick],  self.timestamps['Start'],  self.timestamps['Stop'], 3, 1)
        for neuron in self.neurons:
            if len(spont_times) > 0:  # only run if there were spontaneous intervals
                spontbins = []
                for i in range(0, len(spont_times)):  # for each interval
                    # get all if the neuron spikes for this interval
                    spont_spikes = self.timestamps[neuron][np.where((self.timestamps[neuron] >= spont_times[i][0]) & (
                            self.timestamps[neuron] < spont_times[i][1]))[0]] - spont_times[i][0]
                    # bin the spikes and add to the list
                    spontbins.extend(np.divide(np.histogram(spont_spikes, range(0, math.floor(
                        (spont_times[i][1] - spont_times[i][0]) / 1) * spontbin + 1, spontbin))[0],
                                               spontbin).tolist())
            # get the mean and STD of the spontaneous spike rate
            self.all_spont.append({'File Name': self.filename, 'Neuron': neuron, 'Mean Spont': np.mean(
                spontbins), 'SD Spont': np.std(spontbins, ddof=1)})
        self.final_df['spont'] = pd.DataFrame(self.all_spont)

    def baseline(self):
        for neuron in self.neurons:
            baselines = []
            for key, value in self.trial_times.items():  # for each tastant
                if 'lxl' in key:  # skip if drylick
                    continue
                allspikes = []
                trial_count = len(value)
                for trial_time in value:  # for each trial, get spikes centered around trial time
                    allspikes.extend((self.timestamps[neuron][np.where((self.timestamps[neuron] > trial_time - 2) & (
                            self.timestamps[neuron] < trial_time + 4))] - trial_time).tolist())
                allbins = np.histogram(allspikes, np.linspace(-2, 4, num=int(
                    (4 + 2) / 0.1 + 1)))[0]  # create histogram of spikes
                baselines.append(np.mean(allbins[0:20]) / (0.1 * trial_count))
            self.all_baseline.append({'File Name': self.filename, 'Neuron': neuron, 'Baseline Mean': np.mean(
                baselines), 'Baseline STDEV': np.std(baselines, ddof=1)})
        self.final_df['baseline'] = pd.DataFrame(self.all_baseline)

    def chisquareLXL(self):
        for neuron in self.neurons:
            LXL_data = {}
            dryspikes = []
            # get all spike timestamps around drylick activity, time adjusted around each drylick
            for i in range(len(self.trial_times[self.drylick+'_lxl'])):
                dryspikes.extend(self.timestamps[neuron][np.where((self.timestamps[neuron] >= self.trial_times[self.drylick+'_lxl'][i]) & (
                    self.timestamps[neuron] < (self.trial_times[self.drylick+'_lxl'][i]+0.150)))] - self.trial_times[self.drylick+'_lxl'][i])
            # number of drylick trial-related neuron spikes
            ndry = len(dryspikes)
            dryhist = np.histogram(dryspikes, np.arange(
                0, 0.150+0.015, 0.015))[0]  # bin the spikes
            for tastant in self.tastants:  # run chi square
                lxl_datum = self.LXL(neuron, tastant, self.timestamps[tastant], self.timestamps[neuron], dryhist, ndry, lick_max=0.150,
                                lick_window=0.015, wil_alpha=0.05, chi_alpha=0.05)
                self.all_LXL.append(lxl_datum)

        self.final_df['lxl'] = pd.DataFrame(self.all_LXL)

    def changepoint(self):
        file_5L = pd.DataFrame(columns=['File', 'Neuron', 'Stimulus', 'Trial Count', 'Baseline', 'Magnitude',
                                        'Adj. Magnitude', 'Latency', 'Duration', 'CP1', 'CP2', 'CP3', 'CP4', 'Init CPA',
                                        'eff. alpha', 'pen'])

        for neuron in self.neurons:
            # get penalty value for this neuron
            bootpen, eff_a =  self.get_5L_penalty(
                self.timestamps[neuron], 1000, 0.05, 0.005)
            print('Effective 5-Lick alpha for {} is {}.'.format(neuron, eff_a))
            for key, value in self.trial_times.items():  # for each tastant
                if 'lxl' in key:  # skip if drylick
                    continue
                allspikes = []
                trial_count = len(value)
                for trial_time in value:  # for each trial, get spikes centered around trial time
                    allspikes.extend((self.timestamps[neuron][np.where((self.timestamps[neuron] > trial_time - 2) & (
                            self.timestamps[neuron] < trial_time + 4))] - trial_time).tolist())
                allbins = np.histogram(allspikes, np.linspace(-2, 4, num=int(
                    (4 + 2) / 0.01 + 1)))[0]  # create histogram of spikes
                algo = rpt.Pelt(model='l1', min_size=3, jump=1).fit(
                    allbins)  # fit changepoint model
                # predict changepoints based on calculated penalty value
                changepoints = algo.predict(pen=bootpen)
                changepoints = np.array(changepoints)
                # get change times (in seconds relative to stimulus onset)
                change_times = (np.asarray(changepoints[:-1]) - 20) / 10
                init = np.copy(change_times)
                # check if any changepoints were close to 0 (in the negative) and
                # moving it to 0 would not put it too close to another
                if np.any((change_times < 0) & (change_times >= -0.3)) and \
                        not np.any((change_times >= 0) & (change_times < 0.3)):
                    changepoints[np.where((change_times < 0) & (change_times > -0.3))[0]] = 20
                    change_times[np.where((change_times < 0) & (change_times > -0.3))[0]] = 0
                changepoints = np.delete(changepoints, np.where(change_times < 0)[0])[:-1]
                change_times = np.delete(change_times, np.where(change_times < 0)[0])
                baseline = np.mean(allbins[0:20]) / (0.1 * trial_count)
                this_5L = {'File': self.filename,
                           'Neuron': neuron,
                           'Stimulus': key,
                           'Trial Count': trial_count,
                           'Baseline': baseline,
                           'Magnitude': 'nan',
                           'Adj. Magnitude': 'nan',
                           'Latency': 'nan',
                           'Duration': 'nan',
                           'CP1': 'nan',
                           'CP2': 'nan',
                           'CP3': 'nan',
                           'CP4': 'nan',
                           'Init CPA': init,
                           'eff alpha': eff_a,
                           'pen': bootpen}
                # If there was a response
                if not (len(change_times) == 1 and change_times[0] > 3.5 or len(change_times) == 0):
                    if len(change_times) == 1:  # for one change point
                        respbins = allbins[changepoints[0]:]
                        this_5L['Duration'] = 4 - change_times[0]
                    else:  # for multiple change points
                        respbins = allbins[changepoints[0]:changepoints[1]]
                        this_5L['Duration'] = change_times[1] - change_times[0]
                    # getting the rest of the stats
                    this_5L['Magnitude'] = np.mean(respbins) / (0.1 * trial_count)
                    this_5L['Adj. Magnitude'] = this_5L['Magnitude'] - baseline
                    this_5L['Latency'] = change_times[0]
                    this_5L['Trial Count'] = trial_count
                    for n in range(len(change_times)):
                        if len(change_times) <= 4:
                            this_5L['CP' + str(n + 1)] = change_times[n]
                        else:
                            this_5L['CP' + str(n + 1)] = '{} change points found'.format(
                                len(change_times))
                file_5L = file_5L.append(this_5L, ignore_index=True)
        self.all_5L.append(file_5L)
        self.final_df['5L'] = pd.DataFrame(self.all_5L)

    def coherence(self):
        binsize, numbin = 1 / (2 * 50), 2 * 256
        coh_data = pd.DataFrame(columns=['File', 'Neuron', 'coh peak (4-9Hz)', 'coh peak (6-8Hz)',
                                         'coh peak (10-15Hz)', 'coh mean (4-9)', 'coh mean (6-8)', 'coh mean (10-15)'])
        # bin size and number of bins per segment
        lick_bin = np.histogram(self.timestamps[self.lick], np.arange(
            self.timestamps['Start'], self.timestamps['Stop'], self.params.binsize))[0]  # binning licks
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
            cell_bin = np.histogram(self.timestamps[neuron], np.arange(self.timestamps['Start'], self.timestamps['Stop'], binsize))[0]

            f, Cxy = signal.coherence(lick_bin, cell_bin, fs=1 / binsize, nperseg=numbin,
                                      window='hann', detrend=False, noverlap=int(numbin / 2))
            # get coherence ranges
            four, six, ten = Cxy[np.where((f > 4) & (f < 9))], Cxy[np.where(
                (f > 6) & (f < 8))], Cxy[np.where((f > 10) & (f < 15))]
            coherent = False
            # if the user specified number of consecutive bins are significant
            for i in range(0, len(four) - 3):
                if np.all(four[i:i + 3] > Z):
                    coherent = True
            if coherent:  # record coherence data
                coh_data.loc[len(coh_data)] = [self.filename, neuron, max(four), max(
                    six), max(ten), np.mean(four), np.mean(six), np.mean(ten)]
            else:  # if not coherent, nan
                coh_data = coh_data.append(
                    [pd.Series(dtype='float64')], ignore_index=True)
                coh_data.loc[len(coh_data) - 1, 0:2] = [self.filename, neuron]
        self.all_COH.append(coh_data)
        # self.final_df['COH'] = pd.DataFrame(self.all_COH)

    def ili(self):
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
                                                        np.where((self.timestamps[tastant] >= self.trial_times[tastant][i]) & (
                                                                self.timestamps[tastant] < self.trial_times[tastant][i + 1]))[0][
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
        self.all_ILI.append(ILIs)

        allILIdata = pd.DataFrame(index=self.tastants,
                                  columns=['Median ILI', 'Pause Count', 'Mean Pause Length', 'Mean Pause Length SEM',
                                           'Median Pause Length', 'Mean Time to Complete 5L',
                                           'Mean Time to Complete 5L SEM', 'Median Time to Complete 5L', 'Trial Count'])
        allILIs = {}
        for tastant in self.tastants:
            tastant_ILIs = []
            tastant_5L_times = []
            trial_count = 0
            for fileili in self.all_ILI:  # compile ILI info across all sessions
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

    def get_5L_penalty(self, neur, bootstrap_n=1000,
                       boot_alpha=.05, alpha_tol=.005):
        """Obtain penalty value for us in change-point calculation."""
        bootbins = []
        for key, value in self.trial_times.items():  # for each tastant
            if 'lxl' in key:  # skip if drylick
                continue
            allspikes = []
            # for each trial, collect the spikes, normalized for trial time
            for trial_time in value:
                allspikes.extend((neur[np.where(
                    (neur > trial_time - 2) & (neur < trial_time + 4))] - trial_time).tolist())
            # create histogram of spikes
            allbins = np.histogram(allspikes, np.linspace(-2, 4, num=int((4 + 2) / 0.1 + 1)))[0]
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
            pelts.append(rpt.Pelt(model='l1', min_size=3, jump=1).fit(
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
                old_bootpen = bootpen  # save old bootpen in case the new one overshoots
                bootpen += (eff_a - boot_alpha) * 100 * scale / eff_a # increase the penalty value
        return bootpen, eff_a

    def LXL(self, neuron, tastant, stim, neur, dryhist, ndry, lick_max=.150,
            lick_window=.015, wil_alpha=.05, chi_alpha=.05):
        """Run a lick-by-lick taste response calculation."""
        respmag = None
        tastespikes = []
        # get all spike self.timestamps around tastant activity,
        # time adjusted around each tastant lick
        for i in range(len(stim)):
            tastespikes.extend(neur[np.where((neur >= stim[i]) & (neur < (stim[i] + lick_max)))] - stim[i])
        ntaste = len(tastespikes)
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

        if q_value < chi_alpha:
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
    def binmerge(self, expected, observed):
        """ Merges bins in the case where bin counts would invalidate chi square. """
        e = np.array(expected)
        o = np.array(observed)
        if len(np.where(e < 1)[0]) > 0:  # if any bins are less than one
            for this_bin in np.where(e < 1)[0]:  # merge each bin with the following bin
                if this_bin == len(e) - 1: this_bin = this_bin - 1  # if the last bin move counter back one
                e[this_bin + 1], o[this_bin + 1] = e[this_bin + 1] + e[this_bin], o[this_bin + 1] + o[this_bin]  # merge bins
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
        spont_starts = licks[spont_indices]+prespont
        spont_ends = licks[spont_indices+1]-postspont  # end times
        # get the interval for the beginning, if there is one
        if licks[0]-start[0] >= 10:
            spont_starts = np.sort(np.append(spont_starts, 0))
            spont_ends = np.sort(np.append(spont_ends, licks[0]-postspont))
        # get the interval for the end, if there is one
        if stop[0]-licks[-1] >= 10:
            spont_starts = np.sort(np.append(spont_starts, licks[-1]+prespont))
            spont_ends = np.sort(np.append(spont_ends, stop[0]))
        # combine the start and end times into a tuple of tuples
        spont_times = tuple(zip(spont_starts, spont_ends))
        return spont_times