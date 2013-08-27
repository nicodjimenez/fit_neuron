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


def find_corner_spikes(t, train, ibegin, ti, te):
    """
    Return the times (t1,t2) of the spikes in train[ibegin:]
    such that t1 < t and t2 >= t.
    """
    if(ibegin == 0):
        tprev = ti
    else:
        tprev = train[ibegin-1]
    for idts, ts in enumerate(train[ibegin:]):
        if(ts >= t):
            return np.array([tprev, ts]), idts+ibegin
        tprev = ts
    return np.array([train[-1],te]), idts+ibegin

def bivariate_spike_distance(t1, t2, ti, te, N):
    """
    TODO: test this function 
    
    This Python code (including all further comments) was written by Jeremy Fix (see http://jeremy.fix.free.fr/),
    based on Matlab code written by Thomas Kreuz.
    
    The SPIKE-distance is described in this paper:
    
    .. [KT2013] Kreuz T, Chicharro D, Houghton C, Andrzejak RG, Mormann F:
                Monitoring spike train synchrony.
                J Neurophysiol 109, 1457-1472 (2013).

    Computes the bivariate SPIKE distance of Kreuz et al. (2012)
    
    :param t1: 1D array with the spiking times of two neurons.
    :param t2: 1D array with the spiking times of two neurons.   
    :param ti: beginning of time interval.
    :param te: end of time intervals.
    :param N: number of samples.
    :returns: Array of the values of the distance between time ti and te with N samples.
       
    .. note::
        The arrays t1, t2 and values ti, te are unit less 
    """
    t = np.linspace(ti+(te-ti)/N, te, N)
    d = np.zeros(t.shape)

    t1 = np.insert(t1, 0, ti)
    t1 = np.append(t1, te)
    t2 = np.insert(t2, 0, ti)
    t2 = np.append(t2, te)

    # We compute the corner spikes for all the time instants we consider
    # corner_spikes is a 4 column matrix [t, tp1, tf1, tp2, tf2]
    corner_spikes = np.zeros((N,5))
 
    ibegin_t1 = 0
    ibegin_t2 = 0
    corner_spikes[:,0] = t
    for itc, tc in enumerate(t):
        corner_spikes[itc,1:3], ibegin_t1 = find_corner_spikes(tc, t1, ibegin_t1, ti, te)
        corner_spikes[itc,3:5], ibegin_t2 = find_corner_spikes(tc, t2, ibegin_t2, ti, te)

    #print corner_spikes
    xisi = np.zeros((N,2))
    xisi[:,0] = corner_spikes[:,2] - corner_spikes[:,1]
    xisi[:,1] = corner_spikes[:,4] - corner_spikes[:,3]
    norm_xisi = np.sum(xisi,axis=1)**2.0

    # We now compute the smallest distance between the spikes in t2 and the corner spikes of t1
    # with np.tile(t2,(N,1)) we build a matrix :
    # np.tile(t2,(N,1)) =    [   t2   ]        -   np.tile(reshape(corner_spikes,(N,1)), t2.size) = [                        ]
    #                        [   t2   ]                                                             [  corner  corner  corner]
    #                        [   t2   ]                                                             [                        ]
    dp1 = np.min(np.fabs(np.tile(t2,(N,1)) - np.tile(np.reshape(corner_spikes[:,1],(N,1)),t2.size)),axis=1)
    df1 = np.min(np.fabs(np.tile(t2,(N,1)) - np.tile(np.reshape(corner_spikes[:,2],(N,1)),t2.size)),axis=1)
    # And the smallest distance between the spikes in t1 and the corner spikes of t2
    dp2 = np.min(np.fabs(np.tile(t1,(N,1)) - np.tile(np.reshape(corner_spikes[:,3],(N,1)),t1.size)),axis=1)
    df2 = np.min(np.fabs(np.tile(t1,(N,1)) - np.tile(np.reshape(corner_spikes[:,4],(N,1)),t1.size)),axis=1)

    xp1 = t - corner_spikes[:,1]
    xf1 = corner_spikes[:,2] - t 
    xp2 = t - corner_spikes[:,3]
    xf2 = corner_spikes[:,4] - t

    S1 = (dp1 * xf1 + df1 * xp1)/xisi[:,0]
    S2 = (dp2 * xf2 + df2 * xp2)/xisi[:,1]

    d = (S1 * xisi[:,1] + S2 * xisi[:,0]) / (norm_xisi/2.0)

    return t,d

def multivariate_spike_distance(spike_trains, ti, te, N):
    ''' t is an array of spike time arrays
    ti the initial time of the recordings
    te the end time of the recordings
    N the number of samples used to compute the distance
    spike_trains is a list of arrays of shape (N, T) with N spike trains
    The multivariate distance is the instantaneous average over all the pairwise distances
    '''
    d = np.zeros((N,))
    n_trains = len(spike_trains)
    t = 0
    for i, t1 in enumerate(spike_trains[:-1]):
        for t2 in spike_trains[i+1:]:
            tij, dij = bivariate_spike_distance(t1, t2, ti, te, N)
            if(i == 0):
                t = tij # The times are only dependent on ti, te, and N
            d = d + dij
    d = d / float(n_trains * (n_trains-1) /2)
    return t,d

def test_biv_spk():
    import matplotlib.pylab as plt
    # We test the simulation of Kreuz(2012)

    ######################
    # With 2 spikes trains
    ti = 0
    tf = 1300
    t1 = np.arange(100, 1201, 100)
    t2 = np.arange(100, 1201, 110)
    t, Sb = bivariate_spike_distance(t1, t2, ti, tf, 50)

    plt.figure(figsize=(20,6))

    plt.subplot(211)
    for i in range(t1.size):
        plt.plot([t1[i], t1[i]], [0.5, 1.5], 'k')
    for i in range(t2.size):
        plt.plot([t2[i], t2[i]], [1.5, 2.5], 'k')
    plt.xlim([ti,tf])
    plt.ylim([0,3])
    plt.title("Spike trains")

    plt.subplot(212)
    plt.plot(t, Sb,'k')
    plt.xlim([ti,tf])
    plt.ylim([0,1])
    plt.xlabel("Time (ms)")
    plt.title("Bivariate SPIKE distance")

    plt.savefig("kreuz_bivariate.png")

    plt.show()

    #############################
    # With multiple spikes trains
    ti = 0
    tf = 4000
    num_trains = 50
    num_spikes = 40 # Each neuron fires exactly 40 spikes
    num_events = 5  # Number of events with increasing jitter
    # spike_times is an array where each rows contains the spike times of a neuron
    spike_times = np.zeros((num_trains, num_spikes))
    # The first spikes are randomly spread in the first half of the simulation time
    spike_times[:,range(num_spikes/2)] = tf/2.0 * np.random.random((num_trains, num_spikes/2))
    # We now append the times for the events with increasing jitter
    for i in range(1,num_events+1):
        tb = tf/2.0 * i / num_events 
        spike_times[:,num_spikes/2+i-1] = tb + (50 *(i-1) / num_events)* (2.0 * np.random.random((num_trains,)) - 1)

    # And the second events with the decreasing jitter
    num_last_events = num_spikes/2-num_events
    for i in range(num_last_events):
        tb = tf/2.0 + tf/2.0 * i / (num_last_events-1)
        spike_times[:, -(i+1)] = tb + (50 - 50 *i / (num_last_events-1))* (2.0 * np.random.random((num_trains,)) - 1)

    # Finally we sort the spike times by increasing time for each neuron
    spike_times.sort(axis=1) 
    
    # We compute the multivariate SPIKE distance
    list_spike_trains = []
    [list_spike_trains.append(spike_times[i,:]) for i in range(num_trains)]
    t, Sb = multivariate_spike_distance(list_spike_trains, ti, tf, 1000)

    plt.figure(figsize=(20,6))
    plt.subplot(211)
    # We plot the spike trains
    for i in range(spike_times.shape[0]):
        for j in range(spike_times.shape[1]):
            plt.plot([spike_times[i][j], spike_times[i][j]],[i, i+1],'k')
    plt.title('Spikes')
    plt.subplot(212)
    plt.plot(t,Sb,'k')
    plt.xlim([0, tf])
    plt.ylim([0, 1])
    plt.xlabel("Time (ms)")
    plt.title("Multivariate SPIKE distance")

    plt.savefig("kreuz_multivariate.png")

    plt.show()
    
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
