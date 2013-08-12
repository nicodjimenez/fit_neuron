"""
This module contains class definitions for raw voltage trace data 
and processed voltage trace data that facilitates 
the estimation of LIF models.  This class is not meant for generic use, as it is
designed specifically for the estimation produces in :mod:`fit_neuron.optimize`.  
Processed data means that the indices corresponding to times during which 
the neuron was spiking have been removed.  See :func:`raw_2_processed` on 
details of how this is done.  

A *sweep* is a single pair of input current injections and membrane voltage.
"""

import numpy as np

class MySweep():
    """
    Generic class used by ProcessedData and RawData iterators (see :meth:`ProcessedData.__getitem__`).
    Syntactic sugar for iterating over sweeps.  
    """
    def __init__(self,**kwargs):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
            
class ProcessedData():
    """
    Basic unit of processed data containing multiple sweeps for a single neuron. 
    
    :param dt: time step (seconds) 
    :param input_current_list: list of input current injection arrays
    :param membrane_voltage_list: list of membrane voltage arrays
    :param reset_ind_list: list of reset index arrays
    :raises: ValueError, raised if the lengths of input lists are not equal to each other.
   
    Clients can iterate over sweeps as follows: 
     
    >>> x = np.zeros( (10) )
    >>> y = np.ones( (10) )
    >>> ind_arr = np.array( [3,6,9] )
    >>> for sweep in ProcessedData(0.0001,[x,x],[y,y],[ind_arr,ind_arr]): 
    ...     print sweep.input_current
    ...     print sweep.membrane_voltage
    ...     print sweep.reset_ind
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
    [3 6 9]
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
    [3 6 9]

    """
    def __init__(self,dt=0.0001,
                 input_current_list=[],
                 membrane_voltage_list=[],
                 reset_ind_list=[]):
        
        if len(membrane_voltage_list) != len(input_current_list):
            raise ValueError("Number of voltage traces does not match number of current injections!")
        elif len(membrane_voltage_list) != len(reset_ind_list):
            raise ValueError("Number of voltage traces does not match number of spike trains!")

        self.dt = dt 
        self.sweep_ct = len(input_current_list)
        self.input_current_list = input_current_list
        self.membrane_voltage_list = membrane_voltage_list
        self.reset_ind_list = reset_ind_list
        self.sweep_fields = ["input_current_list","membrane_voltage_list","reset_ind_list"]
        self.sweep_return_fields = ["input_current","membrane_voltage","reset_ind"]
        
    def __getitem__(self,ind):
        sweep = MySweep(**dict((self.sweep_return_fields[ind_0], getattr(self,self.sweep_fields[ind_0])[ind] ) for ind_0 in range(len(self.sweep_fields)))) 
        return sweep
      
class RawData():
    """
    Basic unit of raw data containing multiple sweeps for a single neuron. 
    
    :param dt: time step (seconds) 
    :param input_current_list: list of input current injection arrays
    :param membrane_voltage_list: list of membrane voltage arrays
    :param reset_ind_list: list of reset index arrays
    :raises: ValueError, raised if the lengths of input lists are not equal to each other.
   
    Clients can iterate over sweeps as follows: 
     
    >>> x = np.zeros( (10) )
    >>> y = np.ones( (10) )
    >>> for sweep in RawData(0.0001,[x,x],[y,y]): 
    ...     print sweep.input_current
    ...     print sweep.membrane_voltage
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
    """
    def __init__(self,dt=0.0001,
                 input_current_list=[],
                 membrane_voltage_list=[]):
        
        if len(membrane_voltage_list) != len(input_current_list):
            raise ValueError("Number of voltage traces does not match number of current injections!")
        
        self.dt = dt 
        self.sweep_ct = len(input_current_list)
        self.input_current_list = input_current_list
        self.membrane_voltage_list = membrane_voltage_list
        self.sweep_fields = ["input_current_list","membrane_voltage_list"]
        self.sweep_return_fields = ["input_current","membrane_voltage"]
        
    def __getitem__(self,ind):
        sweep = MySweep(**dict((self.sweep_return_fields[ind_0], getattr(self,self.sweep_fields[ind_0])[ind] ) for ind_0 in range(len(self.sweep_fields)))) 
        return sweep
    
def raw_2_processed(raw_data,look_behind=0.002):
    """
    Takes raw data, removes the indices of the spikes, and returns the processed data.
    
    Spikes are removed as follows:
     
        1) Use upward zero crossing criterion to find spike time.
        2) Look a fixed amound of time (determined by :attr:`look_behind`)
           before the spike time computed in step 1 and record the value of the voltage here.
           The spike is considered to have started here.
        3) Consider the spike to have ended after the neuron's voltage drops 
           to a value that is below the pre-spike voltage value computed in step 2.  
        4) Remove chunk of data between endpoints determined by step 2 and step 3.
    
    :param raw_data: raw data 
    :type raw_data: :class:`RawData`
    :param look_behind: controls how much we remove from each spike (see above procedure).
    :rtype: :class:`ProcessedData`
    
    .. note::
        The spike shapes are removed not only from the voltage traces, but are 
        also removed from the corresponding indices of the current injections.
        
    """
    
    reset_ind_list = []
    membrane_voltage_list = []
    input_current_list = []
    
    dt = raw_data.dt
    pre_spike_ind = int(look_behind / dt) 
    
    for sweep in raw_data:
        
        input_array = sweep.input_current
        output_array = sweep.membrane_voltage
        reset_ind = []
        ind = 0
        
        while ind < len(input_array):   
            
            if output_array[ind] > 0: 
                departure_ind = ind - pre_spike_ind
                pre_thresh_volt = output_array[departure_ind]
                return_ind = ind
                
                # if the voltage returns to original value look_behind seconds before
                # the spike, exits loop
                while output_array[return_ind] > pre_thresh_volt and ind < len(output_array) - 1:
                    return_ind += 1 
                    
                ind = return_ind 
                ind_delete_list = range(departure_ind,return_ind)
                
                # now we remove the spikes 
                input_array = np.delete(input_array,ind_delete_list)
                output_array = np.delete(output_array,ind_delete_list)
                reset_ind.append(ind_delete_list[0])    
                
            ind += 1 
        
        input_array = input_array[0:ind]
        output_array = output_array[0:ind]
        
        reset_ind_list.append(np.array(reset_ind))
        
        # replace following arrays with processed arrays
        membrane_voltage_list.append(output_array)
        input_current_list.append(input_array)
        
     
    processed_data = ProcessedData(dt=dt,
                                   input_current_list=input_current_list,
                                   membrane_voltage_list=membrane_voltage_list,
                                   reset_ind_list=reset_ind_list)
    
    return processed_data

if __name__ == '__main__':
    import doctest
    doctest.testmod()
        