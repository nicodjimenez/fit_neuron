"""
spkd_lib
~~~~~~~~~~~~~
This module is a library of spike distance metrics. 
For more information about the spike distance metrics 
used below, see http://www.scholarpedia.org/article/Measures_of_spike_train_synchrony.
"""

import sys
import os 
import numpy as np 
from scipy import stats
import sys
from numpy import *
from numpy import floor
from numpy import zeros
from numpy import size

def get_gamma_factor(coincidence_count, model_length, target_length, target_rates, delta):
    NCoincAvg = 2 * delta * target_length * target_rates
    norm = .5 * (1 - 2 * delta * target_rates)
    gamma = (coincidence_count - NCoincAvg) / (norm * (target_length + model_length))
    return gamma

# First-order statistics
def firing_rate(spikes):
    '''
    Rate of the spike train.
    '''
    if spikes==[]:
        return NaN
    return (len(spikes) - 1) / (spikes[-1] - spikes[0])

def gamma_factor(source, target, delta, normalize=True, dt=None):
    """
    Returns the gamma precision factor between source and target trains,
    with precision delta.  See [RJ2008]_ for a more detailed description.
    If normalize is True, the function returns the normalized gamma factor 
    (less than 1.0), otherwise it returns the number of coincidences.
    dt is the precision of the trains, by default it is defaultclock.dt
    
    :param source: array of spike times 
    :param target: array of spike times 
    :param delta: precision of coincidence detector
    
    .. [RJ2008] Jolivet, Renaud, et al. "A benchmark test for a quantitative assessment of simple neuron models." 
        Journal of neuroscience methods 169.2 (2008): 417-424.
    """

    source = array(source)
    target = array(target)
    target_rate = firing_rate(target) 

    if dt is None:
        delta_diff = delta
    else:
        source = array(rint(source / dt), dtype=int)
        target = array(rint(target / dt), dtype=int)
        delta_diff = int(rint(delta / dt))

    source_length = len(source)
    target_length = len(target)

    if (target_length == 0 or source_length == 0):
        return 0

    if (source_length > 1):
        bins = .5 * (source[1:] + source[:-1])
        indices = digitize(target, bins)
        diff = abs(target - source[indices])
        matched_spikes = (diff <= delta_diff)
        coincidences = sum(matched_spikes)
    else:
        indices = [amin(abs(source - target[i])) <= delta_diff for i in xrange(target_length)]
        coincidences = sum(indices)

    # Normalization of the coincidences count
#    NCoincAvg = 2 * delta * target_length * target_rate
#    norm = .5*(1 - 2 * target_rate * delta)
#    gamma = (coincidences - NCoincAvg)/(norm*(source_length + target_length))

    # TODO: test this
    gamma = get_gamma_factor(coincidences, source_length, target_length, target_rate, delta)

    if normalize:
        return gamma
    else:
        return coincidences

def schrieber_sim(st_0,st_1,bin_width=0.0001,sigma=0.1,t_extra=0.5):
    """
    Computes Schrieber similarity between two spike trains as described in [SS2003]_.
    
    :param st_0: array of spike times in seconds 
    :param st_1: second array of spike times in seconds 
    :keyword bin_width: precision in seconds over which Gaussian convolution is computed
    :keyword sigma: bandwidth of Gaussian kernel 
    :keyword t_extra: how much more time in seconds after last signal do we keep convolving?   
    
    .. [SS2003] Schreiber, S., et al. "A new correlation-based measure of spike timing reliability." 
        Neurocomputing 52 (2003): 925-931.
    """
    
    smoother_0 = stats.gaussian_kde(st_0,sigma)
    smoother_1 = stats.gaussian_kde(st_1,sigma)
    t_max = max([st_0[-1],st_1[-1]]) + t_extra
    t_range = np.arange(0,t_max,bin_width)
    st_0_smooth = smoother_0(t_range)
    st_1_smooth = smoother_1(t_range)
    sim = stats.pearsonr(st_0_smooth, st_1_smooth)[0]
    return sim 

class ExpDecay():
    """
    Exponentially decaying function with additive method.  
    Useful for efficiently computing Van Rossum distance.
    """
    
    def __init__(self,k=None,dt=0.0001):

        self.sic_val = 0.0 
        self.dt = dt 
        self.k = k 
        self.decay_factor = exp(-dt*k)
    
    def update(self,V=0):
        self.sic_val = self.sic_val * self.decay_factor
        return self.sic_val
    
    def spike(self):
        self.sic_val += 1 
        return self.sic_val
        
    def reset(self):
        self.sic_val = 0 

def van_rossum_dist(st_0,st_1,tc=1000,bin_width=0.0001,t_extra=1):
    """
    Calculates the Van Rossum distance between spike trains
    as defined in [VR2001]_.  Note that the default parameters 
    are optimized for inputs in units of seconds.  
    
    :param st_0: array of spike times for first spike train
    :param st_1: array of spike times for second spike train
    :param bin_width: precision in units of time to compute integral
    :param t_extra: how much beyond max time do we keep integrating until? \
    This is necessary because the integral cannot in practice be evaluated between \
    :math:`t=0` and :math:`t=\infty`.  
    
    .. [VR2001] van Rossum, Mark CW. "A novel spike distance." 
                Neural Computation 13.4 (2001): 751-763.
    """
    
    # by default, we assume spike times are in seconds,
    # keep integrating up to 0.5 s past last spike         
    t_max = max([st_0[-1],st_1[-1]]) + t_extra
    
    #t_min = min(st_0[0],st_0[0])
    t_range = np.arange(0,t_max,bin_width)
    
    # we use a spike induced current to perform the computation
    sic = ExpDecay(k=1.0/tc,dt=bin_width)
    
    f_0 = t_range * 0.0
    f_1 = t_range * 0.0 
    
    # we make copies of these arrays, since we are going to "pop" them
    s_0 = list(st_0[:])
    s_1 = list(st_1[:])
    
    for (st,f) in [(s_0,f_0),(s_1,f_1)]:
        # set the internal value to zero
        sic.reset()
        
        for (t_ind,t) in enumerate(t_range):
            f[t_ind] = sic.update()
            if len(st)>0:
                if t > st[0]:
                    f[t_ind] = sic.spike()
                    st.pop(0)
                    
    d = np.sqrt((bin_width / tc)* np.linalg.norm((f_0-f_1),1))
    return d 
                 
def victor_purpura_dist(tli,tlj,cost=1):
    """
    d=spkd(tli,tlj,cost) calculates the "spike time" distance
    as defined [DA2003]_ for a single free parameter, 
    the cost per unit of time to move a spike.
    
    :param tli: vector of spike times for first spike train
    :param tlj: vector of spike times for second spike train
    :keyword cost: cost per unit time to move a spike
    :returns: spike distance metric 
    
    Translated to Python by Nicolas Jimenez from Matlab code by Daniel Reich.
    
    .. [DA2003] Aronov, Dmitriy. "Fast algorithm for the metric-space analysis 
                of simultaneous responses of multiple single neurons." Journal 
                of Neuroscience Methods 124.2 (2003): 175-179.
    
    Here, the distance is 1 because there is one extra spike to be deleted at 
    the end of the the first spike train:
    
    >>> spike_time([1,2,3,4],[1,2,3],cost=1)
    1 
    
    Here the distance is 1 because we shift the first spike by 0.2, 
    leave the second alone, and shift the third one by 0.2, 
    adding up to 0.4:
    
    >>> spike_time([1.2,2,3.2],[1,2,3],cost=1)
    0.4

    Here the third spike is adjusted by 0.5, but since the cost 
    per unit time is 0.5, the distances comes out to 0.25:  
    
    >>> spike_time([1,2,3,4],[1,2,3,3.5],cost=0.5)
    0.25
    """
    
    nspi=len(tli)
    nspj=len(tlj)

    if cost==0:
        d=abs(nspi-nspj)
        return d
    elif cost==np.Inf:
        d=nspi+nspj;
        return d

    scr = np.zeros( (nspi+1,nspj+1) )

    # INITIALIZE MARGINS WITH COST OF ADDING A SPIKE

    scr[:,0] = np.arange(0,nspi+1)
    scr[0,:] = np.arange(0,nspj+1)
           
    if nspi and nspj:
        for i in range(1,nspi+1):
            for j in range(1,nspj+1):
                scr[i,j] = min([scr[i-1,j]+1, scr[i,j-1]+1, scr[i-1,j-1]+cost*abs(tli[i-1]-tlj[j-1])])
        
    d=scr[nspi,nspj]
    return d
    
def test():
    st_0 = [1,2,3,4]
    st_1 = [1,2,3,4.002]
    d1 = van_rossum_dist(st_0,st_1,tc=1000)
    d2 = victor_purpura_dist(st_0,st_1)
    #(source, target, delta, normalize=True, dt=None):
    d3 = gamma_factor(st_0,st_1,0.004)
    d4 = schrieber_sim(st_0,st_1,sigma=0.1)
    print "Spike train 1: " + str(st_0)
    print "Spike train 2: " + str(st_1)
    print "Van rossum distance: " + str(d1)
    print "Victor Purpura distance: " + str(d2)
    print "Gamma factor: " + str(d3)
    print "Schrieber similarity: " + str(d4)
    
    
if __name__ == '__main__':
    test()
