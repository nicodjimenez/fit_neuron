'''
Wrapper class for gLIF methods.  
'''

import numpy as np
import os
import pickle
import json
from fit_neuron.data import load_neuron_data
import fit_neuron.optimize
from fit_neuron.optimize import sic_lib

# a default parameter (units of ms) for threshold intervals
T_BIN_DEFAULT = [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0008,0.001,0.00125,0.0015,0.002,0.003,0.004,0.005,0.01,0.015,0.02,0.025,0.03,0.05,0.08,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.7,0.9,1.2]

class gLIF_model():
    def __init__(self,t_bin=T_BIN_DEFAULT):
        """
        We initialize the model.  If parameters are not specified, default parameters 
        will be used that have been found effective in tests.  We should initalize
        a simple model here. 
        """
        
        #:bool reflecting whether the predict method has been called on this object
        self.is_fitted = False
        self.neuron = None
        
    def predict(self,X_list,dt=0.0001):
        """
        Given new input current, predict the output of the neuron.
        """
        
        Y_list = [None] * len(X_list)
        
        if not self.is_fitted: 
            raise RuntimeError("The model has not been fitted! Call obj.fit(...) with proper parameters")
         
        for (ind,input_current) in enumerate(X_list): 
            
            X_elem_len = len(input_current)
            cur_voltage_arr = np.zeros(X_elem_len)
            
            # resets neuron to resting potential
            self.neuron.reset()
            
            for (t_ind,Ie) in enumerate(input_current):
                cur_voltage_arr[t_ind] = self.neuron.update(Ie)
            
            Y_list[ind] = cur_voltage_arr
                      
        return Y_list

    def fit(self,X_list,Y_list,dt=0.0001):
        """
        Fits a gLIF model using default parameters values. A :class:`fit_neuron.optimize.neuron_base_obj.Neuron` instance 
        is returned but is not necessary to use the predicted model.
        
        :param X_list: list of arrays of recorded current injections. 
        :param Y_list: list of arrays of recorded membrane voltages corresponding to :attr:`input_current_list` input.
        :param dt: time increment between consecutive current / voltage values
        :returns: :class:`fit_neuron.optimize.neuron_base_obj.Neuron` instance.
        """
        
        t_bins = T_BIN_DEFAULT
        sic_list = [sic_lib.StepSic(t,dt=dt) for t in t_bins]            

        self.neuron = fit_neuron.optimize.fit_neuron(input_current_list=X_list,
                                                     membrane_voltage_list=Y_list,
                                                     dt=dt,
                                                     process_ct=None,
                                                     max_lik_iter_max=25,
                                                     stopping_criteria=0.01,
                                                     sic_list=sic_list)
        
        self.is_fitted = True
        
        return self.neuron
