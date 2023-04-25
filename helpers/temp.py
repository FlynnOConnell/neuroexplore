def changepoint(self):
    file_5L = pd.DataFrame(columns=['File', 'Neuron', 'Stimulus', 'Trial Count', 'Baseline', 'Magnitude',
                                    'Adj. Magnitude', 'Latency', 'Duration', 'CP1', 'CP2', 'CP3', 'CP4', 'Init CPA',
                                    'eff. alpha', 'pen'])

    for neuron in self.neurons:
        # get penalty value for this neuron
        bootpen, eff_a = self.get_5L_penalty(
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
                       'pen': bootpen
                       }
            # If there was a response
            if not (len(change_times) == 1 and change_times[0] > 3.5 or len(change_times) == 0):
                self.all_responses['5L'] = True
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