"""
extract_spikes 
~~~~~~~~~~~~~~~~~~~~~~

This module contains functions to extract spike times and spike
shapes from voltage traces. 
These voltage traces may be recorded voltage traces, or they 
may be simulated voltage traces.  The convention used here 
is that simulated voltage traces are :attr:`numpy.nan` valued whenever 
the neuron is spiking.   
""" 

import numpy as np

def spk_from_sim(membrane_voltage_list):
    """
    Takes a list of simulated voltage traces and returns list of spike indices.
    This function assumes convention that the voltage is nan corresponds to 
    when the neuron is spiking.
    
    :param membrane_voltage_list: list of arrays of voltage traces
    :returns: list of spike index arrays
    
    .. note:: 
        In order to get the spike *times*, the returned spike index arrays 
        will need to be multiplied by dt.  This step will to be done in 
        order to call functions from :mod:`fit_neuron.evaluate.spkd_lib`.
    """
    
    spk_ind_list = []
    
    for voltage_arr in membrane_voltage_list:
        
        cur_ind_list = []
        is_spiking = False
        
        for t in range(len(voltage_arr)):
            V = voltage_arr[t]
            if np.isnan(V):
                if not is_spiking: 
                    cur_ind_list.append(t)
                    is_spiking = True            
            else: 
                if is_spiking:
                    is_spiking = False 
        
        spk_ind_list.append(np.array(cur_ind_list))
    
    return spk_ind_list

def spk_from_bio(membrane_voltage_list):
    """
    Computes list of arrays of spike indices from list of voltage traces.
    The spike indices are computed using the upward zero crossing criterion. 
    
    :param membrane_voltage_list: list of arrays with units of mV
    :returns: list of arrays of spike indices
    
    .. note:: 
        In order to get the spike *times*, the returned spike index arrays 
        will need to be multiplied by dt.  This step will need to be done in 
        order to call functions from :mod:`fit_neuron.evaluate.spkd_lib`.
    """
    
    spike_ind_raw = []
    
    for cur_ind in range(len(membrane_voltage_list)):

        output_array = membrane_voltage_list[cur_ind]
        arr_len = len(output_array)
        
        # note: we do a reset the index AFTER the spikes 
        spike_ind = []            
        ind = -1 
        
        while ind < arr_len - 1:          
            ind += 1
                         
            if output_array[ind] > 0: 
                #above_thresh_sample = output_array[ind:ind+25]
                #top_ind = above_thresh_sample.argmax() + ind
                spike_ind.append(ind)    
                
                # we keep pushing the index until the voltage comes back 
                # down, this ensures that we don't double count spikes 
                #ind = top_ind
                
                while output_array[ind] > -10 and ind < arr_len - 1: 
                    ind += 1  
                        
        spike_ind_arr = np.array(spike_ind)
        spike_ind_raw.append(spike_ind_arr)
        
    return spike_ind_raw

def compute_spike_shapes(raw_data,look_behind=0.003,look_ahead=0.025):
    """
    Computes values of self.spike_shapes.  The arguments specify amount
    of time in seconds to look ahead and behind of argmax of spike to 
    include as part of the spike shape.  The list of computed spike shapes 
    are stored in :attr:`spike_shapes`.
    
    :keyword look_behind: how much time before the upward zero crossing of the voltage do we consider in spike shape?
    :keyword look_ahead: how much time after the upward zero crossing of voltage do we consider being in spike shape?
    
    .. note:: 
        The parameter look_behind is very important, for it will be used to
        determine the length of the spikes.  The length of the spikes 
        will be computed by computing the amount of time it takes for the voltage
        to return to it's prior value at look_behind seconds before the 
        spike event.  
    """
    
    spike_shapes = []
    bin_ahead_ct = int(look_ahead / raw_data.dt)
    bin_behind_ct = int(look_behind / raw_data.dt)
    
    for sweep in raw_data:
        spike_ind = spk_from_bio([sweep.membrane_voltage])[0]
        spike_delimiters = [(ind-bin_behind_ct,ind+bin_ahead_ct) for ind in spike_ind]  
        volt_trace = sweep.membrane_voltage
        
        for d in spike_delimiters: 
            # this 'if' statement makes sure we don't cause a ValueError
            if min(d) > 0 and max(d) < len(volt_trace) - 1:
                spike_shapes.append(volt_trace[d[0]:d[1]])
                
    return spike_shapes

def compute_t_ref(spike_shapes,dt): 
    """
    Returns the refractory period of the neuron by doing a simple
    calculation that computes the amount of time it takes for median 
    spike shape to return to a voltage value that is below
    the median spike shape's voltage a short period before the actual 
    spike.  
    """
    
    #spike_shapes = compute_spike_shapes(raw_data)
    median_spike_shape = np.zeros_like(spike_shapes[0])
    
    for ind in range(len(median_spike_shape)):
        median_spike_shape[ind] = np.median(([a[ind] for a in spike_shapes]))
    
    argmax = median_spike_shape.argmax()
    ind = argmax 
    
    while ind < len(median_spike_shape):
        ind += 1 
        
        if median_spike_shape[ind] < median_spike_shape[0]:
            break
    
    t_ref = (ind - argmax) * dt
    return t_ref

def compute_Vr(spike_shapes,t_ref,dt):
    """
    Finds the median reset voltage of the neuron t_ref seconds 
    after the peak of the spike.  
    
    .. note:: 
        The median value of the reset voltage at the time of the
        reset indices is actually used instead of the average, as
        the average is less robust to outliers.
    """
    
    median_spike_shape = np.zeros_like(spike_shapes[0])
    
    for ind in range(len(median_spike_shape)):
        median_spike_shape[ind] = np.median(([a[ind] for a in spike_shapes]))
    
    argmax_ind = median_spike_shape.argmax()
    t_ref_ind = int(t_ref / dt)
    
    return median_spike_shape[t_ref_ind + argmax_ind]
    