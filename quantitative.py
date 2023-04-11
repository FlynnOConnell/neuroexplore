import numpy as np
import pandas as pd
from scipy import stats, signal


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


def f_test(x, y):
    """Runs f_test to compare variances."""
    x, y = np.array(x), np.array(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    if var_x > var_y:
        f = var_x / var_y
    else:
        f = var_y / var_x
    dfn = x.size - 1  # define degrees of freedom numerator
    dfd = y.size - 1  # define degrees of freedom denominator
    p = 1 - stats.f.cdf(f, dfn, dfd)  # find p-value of F test statistic
    return f, p

def get_spk_bins(trial_times, neur, start, end, binsize):
    """Bins spikes."""
    start = abs(start)
    allspk = []
    for teemay in trial_times:
        allspk.extend((neur[np.where(
            (neur > teemay - start) & (neur < teemay + end))] - teemay).tolist())
    bins = \
        np.histogram(allspk,
                     np.linspace(-start, end, num=int((start + end) / binsize) + 1))[
            0]
    return bins

def binmerge(expected, observed):
    """ Merges bins in the case where bin counts would invalidate chi square. """
    e = np.array(expected)
    o = np.array(observed)
    if len(np.where(e < 1)[0]) > 0:  # if any bins are less than one
        for bin in np.where(e < 1)[0]:  # merge each bin with the following bin
            if bin == len(e) - 1: bin = bin - 1  # if the last bin move counter back one
            e[bin + 1], o[bin + 1] = e[bin + 1] + e[bin], o[bin + 1] + o[
                bin]  # merge bins
            e[bin], o[bin] = 0, 0
        o = o[np.where(e != 0)[0]]  # remove 0 bins
        e = e[np.where(e != 0)[0]]
    while len(np.where(e < 5)[0]) / len(e) > .2 and len(
            e) > 1:  # greater than 80% of bins must be 5 or higher
        bin = np.where(e < 5)[0][0]  # find first offending bin and merge
        if bin == len(e) - 1: bin = bin - 1
        e[bin + 1], o[bin + 1] = e[bin + 1] + e[bin], o[bin + 1] + o[bin]
        e[bin], o[bin] = 0, 0
        o = o[np.where(e != 0)[0]]
        e = e[np.where(e != 0)[0]]
        e = e[np]
    return e, o

def LXL(
        stim, neur, dryhist, ndry, lick_max=.150,
        lick_window=.015, wil_alpha=.05, chi_alpha=.05, drylick=None, tastants=None):
    """Run a lick-by-lick taste response calculation."""
    tastespikes = []
    # get all spike timestamps around tastant activity,
    # time adjusted around each tastant lick
    for i in range(len(stim)):
        tastespikes.extend(
            neur[np.where((neur >= stim[i]) & (neur < (stim[i] + lick_max)))] - stim[i])
    ntaste = len(tastespikes)  # number of taste spikes
    tasteobs = \
        np.histogram(tastespikes, np.arange(0, lick_max + lick_window, lick_window))[
            0]  # bin the tastant spikes
    expected = dryhist * ntaste / ndry
    if sum(expected) >= 10:
        expected, tasteobs = binmerge(expected, tasteobs)
        chi_stat, gof_p = stats.chisquare(tasteobs, f_exp=expected)  # run chi square
        # get firing rates for drylick and tastant
        dryfire = (expected * ndry / ntaste) / len(
            trial_times[drylick + '_lxl']) / lick_window
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
    q_value = gof_p * len(tastants)  # bonferroni correction
    # if the chi square is significant, record the taste response
    if q_value < chi_alpha:
        this_LXL = {
            'File Name'         : file,
            'Neuron'            : neuron,
            'Tastant'           : tastant,
            'Trial Count'       : len(stim),
            'P-Value'           : gof_p,
            'Q-Value'           : q_value,
            'Response Magnitude': respmag,
            'Wilcoxon_p'        : pval,
            'Response_Type'     : resptype}
    else:
        this_LXL = {
            'File Name'         : file,
            'Neuron'            : neuron,
            'Tastant'           : tastant,
            'Trial Count'       : len(stim),
            'P-Value'           : gof_p,
            'Q-Value'           : q_value,
            'Response Magnitude': 'nan',
            'Wilcoxon_p'        : 'nan',
            'Response_Type'     : 'nan'}
    return this_LXL

def spont_intervals(licks, start, stop, prespont, postspont):
    """Determine the intervals for spontaneous activity evaluation."""
    # find ILIs long enough to be a spontaneous interval
    spont_indices = np.where(np.diff(licks) >= spont)[0]
    # spontaneous interval start and end times
    spont_starts = licks[spont_indices] + prespont
    spont_ends = licks[spont_indices + 1] - postspont  # end times
    # get the interval for the beginning, if there is one
    if licks[0] - start[0] >= spont:
        spont_starts = np.sort(np.append(spont_starts, 0))
        spont_ends = np.sort(np.append(spont_ends, licks[0] - postspont))
    # get the interval for the end, if there is one
    if stop[0] - licks[-1] >= spont:
        spont_starts = np.sort(np.append(spont_starts, licks[-1] + prespont))
        spont_ends = np.sort(np.append(spont_ends, stop[0]))
    # combine the start and end times into a tuple of tuples
    spont_times = tuple(zip(spont_starts, spont_ends))
    return spont_times

def get_5L_penalty(
        trial_times, neur, bootstrap_n=1000,
        boot_alpha=.05, alpha_tol=.005):
    """Obtain penalty value for us in change-point calculation."""
    bootbins = []
    for key, value in trial_times.items():  # for each tastant
        if 'lxl' in key:  # skip if drylick
            continue
        allspikes = []
        # for each trial, collect the spikes, normalized for trial time
        for trial_time in value:
            allspikes.extend((neur[np.where((neur > trial_time - base) & (
                    neur < trial_time + trial_end))] - trial_time).tolist())
        # create histogram of spikes
        allbins = np.histogram(allspikes, np.linspace(-base, trial_end, num=int(
            (trial_end + base) / bin5L + 1)))[0]
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
