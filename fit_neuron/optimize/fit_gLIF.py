""" 
fit_gLIF 
~~~~~~~~~~~~~
Fits subthreshold and threshold parameters of a gLIF model from voltage 
patch clamp data.  A neuron object is returned to the client that
can easily be simulated using standard methods listed in the documentation.
"""

#import numpy as np
#from .. import data
#from ..data import NeuronData
import sys 
from fit_neuron import data
from fit_neuron.data import RawData
import threshold
import subthreshold 
import sic_lib 
from neuron_base_obj import Neuron

# this list determines the endpoints of the intervals that are used 
# to specify the shape of the dynamic threshold 
T_BIN_DEFAULT = [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0008,0.001,0.00125,0.0015,0.002,0.003,0.004,0.005,0.01,0.015,0.02,0.025,0.03,0.05,0.08,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.7,0.9,1.2]

# defines time constants of voltage chasing currents
TIME_CONST_DEFAULT = [1]

NONLIN_FCN_DICT_DEFAULT = None 

K_VECTOR_DEFAULT = [10,25,50,75,100,150,200]
            
def fit_neuron(input_current_list=None,
               membrane_voltage_list=None,
               dt=None,
               volt_nonlin_fcn=None,
               sic_list=None,
               t_thresh_bins=[],
               volt_adapt_time_const=[],
               thresh_param_init=None,
               max_lik_iter_max=20,
               process_ct = None, 
               stopping_criteria = 0.01,
               look_behind=0.002
               ):
    r"""
    Fits an actionable neuron to the data provided.
    
    :param input_current_list: list of arrays of recorded current injections. 
    :param membrane_voltage_list: list of arrays of recorded membrane voltages corresponding to :attr:`input_current_list` input.
    :param dt: time step. 
    :param volt_nonlin_fcn: voltage nonlinearity function (see :class:`fit_neuron.optimize.subthreshold.Voltage`).
    :param sic_list: list of spike induced current objects (see :mod:`fit_neuron.optimize.sic_lib`).
    :param t_thresh_bins: list of endpoints of time intervals used to compute step-like threshold (see :class:`fit_neuron.optimize.threshold.StochasticThresh`).
    :param volt_adapt_time_const: list of time constancs :math:`Q_i` for voltage chasing currents. (see :class:`fit_neuron.optimize.threshold.StochasticThresh`).
    :param thresh_param_init: initial value of threshold parameters for max likelihood threshold optimization (see :func:`fit_neuron.optimize.threshold.max_likelihood`). 
    :param make_lik_iter_max: maximum number of max likelihood iterations (see :func:`fit_neuron.optimize.threshold.max_likelihood`). 
    :param process_ct: number of processors we want to use to distribute max likelihood; if None then use max CPU count (see :func:`fit_neuron.optimize.threshold.max_likelihood`).
    :param stopping_criteria: minimum :math:`L^2` norm of log likelihood gradient below which we stop optimization (see :func:`fit_neuron.optimize.threshold.max_likelihood`).
    :param look_behind: parameter that controls how the length of the refractory period is computed (see :func:`fit_neuron.data.my_data.raw_2_processed`).
    :returns: :class:`fit_neuron.optimize.neuron_base_obj.Neuron` instance.
    """
        
    if sic_list == None:
        sic_list = [sic_lib.StepSic(t,dt=dt) for t in T_BIN_DEFAULT]
        
    if len(t_thresh_bins) == 0: 
        t_thresh_bins = T_BIN_DEFAULT 
    
    raw_data = RawData(input_current_list=input_current_list,
                   membrane_voltage_list=membrane_voltage_list,
                   dt=dt)

    
    spike_shapes = data.compute_spike_shapes(raw_data,look_behind=look_behind)
    t_ref = data.compute_t_ref(spike_shapes,dt)
    Vr = data.compute_Vr(spike_shapes,t_ref,dt)
    
    subthresh_obj = subthreshold.Voltage(sic_list=sic_list,
                                         volt_nonlin_fcn=volt_nonlin_fcn, 
                                         dt=dt, 
                                         Vr=Vr, 
                                         t_ref=t_ref)
    
    processed_data = data.raw_2_processed(raw_data,look_behind=look_behind)
    
    subthresh_param_arr = subthreshold.estimate_volt_parameters(subthresh_obj, processed_data)
    
    #print "Subthresh arr: " + str(subthresh_param_arr)
    subthresh_obj.set_param(subthresh_param_arr)
        
    thresh_obj = threshold.StochasticThresh(t_bins=t_thresh_bins,
                                            volt_adapt_time_const=volt_adapt_time_const,
                                            dt=dt)
    
    thresh_param_arr = threshold.estimate_thresh_parameters(subthresh_obj,
                                                            thresh_obj,
                                                            raw_data,
                                                            process_ct=process_ct,
                                                            max_lik_iter_max=max_lik_iter_max,
                                                            thresh_param_init=thresh_param_init,
                                                            stopping_criteria=stopping_criteria,
                                                            )
    thresh_obj.set_param(thresh_param_arr)
    
    neuron = Neuron(subthresh_obj=subthresh_obj,
                    thresh_obj=thresh_obj,
                    V_init=None)
    return neuron

if __name__ == "__main__": 
    pass
    


