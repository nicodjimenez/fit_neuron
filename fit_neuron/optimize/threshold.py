"""
threshold
~~~~~~~~~~~~~~~~~~~~
Parameter extraction of threshold parameters from voltage clamp data.   
"""

import sys
import os 
import numpy as np
import random 
from numpy import linalg
from numpy import exp, Inf, floor, ceil, zeros 
import multiprocessing
from fit_neuron import data

def get_grad_and_hessian((thresh_param,X_cat)):
    """
    Takes a subset of the data and returns gradient and hessian for this
    subset of the data. This function is called by par_calc_log_like_update
    and allows this function to work in parallel, map-reduce style.
    Computes gradient vector as well as Hessian matrix of
    log likelihood w.r.t. threshold parameters, evaluated
    at thresh_param. The grad and hess can then be used to update
    thresh parameters via a Newton-Raphson step.
    """
    theta_len = len(thresh_param)
    thresh_param = thresh_param.reshape((1,theta_len))
    grad_vec = np.zeros( (1,theta_len) )
    hess_mat = np.zeros( (theta_len,theta_len) )
    data_len = len(X_cat)
    
    for t in range(data_len):
        X_thresh = X_cat[t,:].reshape((1,theta_len))
        
        if np.isnan(X_thresh[0,0]):
            continue
        
        if t+1 < data_len:
            if np.isnan(X_cat[t+1,0]):
                grad_vec = grad_vec + X_thresh
                continue
        
        exp_val = np.exp(thresh_param.dot(X_thresh.T))[0][0]
        grad_vec = grad_vec - X_thresh * exp_val
        hess_mat = hess_mat - exp_val * X_thresh.T.dot(X_thresh)

    cur_tuple = (grad_vec,hess_mat)
    return cur_tuple 

def par_calc_log_like_update(thresh_param,X_cat_split,process_ct):
    """
    Parallel version of Newton update. This method is called
    by :func:`max_likelihood`.
    The function splits data into chunks, passes chuncks to
    different processors, collects the result to obtain
    the gradient and hessian of the whole time series, then
    performs Newon update.
    """
    pool = multiprocessing.Pool(processes=process_ct)
    inputs = [(thresh_param[:],X_cat_short) for X_cat_short in X_cat_split]
    
    results = pool.map(get_grad_and_hessian, inputs)
    
    pool.close()
    pool.join()
    
    grad_sum = results[0][0]
    hess_sum = results[0][1]
    for r in results[1:]:
        grad_sum += r[0]
        hess_sum += r[1]
    
    theta_new = thresh_param - np.linalg.pinv(hess_sum).dot(grad_sum.T).T
    return grad_sum,theta_new[0] 

class StochasticThresh():
    r"""
    Class for the threshold component of a spiking neuron.  The main functionality
    of this class is to determine spike times stochastically. 
    
    :param dt: time step.
    :keyword t_bins: a list that defines the :math:`b_i` that define indicator functions :math:`I_{[0,b_i]}(t)` (see below).  
    :keyword volt_adapt_time_const: a list that defines the time constants :math:`r_i` of the voltage chasing currents (see below).
    :keyword t_hist: spiking history of neuron, how long ago were the last spikes? 
    
    The stochastic neuron has the following *hazard rate*: 
    
    .. math:: 
        h(t) = \exp \left({\bf w}_t^{\top} {\bf X}_t (t) \right)
    
    where 
    
    .. math:: 
        {\bf X}_t (t) = [1,V(t),I_1(t),\hdots,I_m(t),Q_1(t),\hdots,Q_l(t)]^{\top}.
    
    where the :math:`I_i(t) = I_{[0,b_i]}(t)` parameters are the indicator variables, and 
    the :math:`Q_j(t)` parameters are probability currents which shall be referred to as 
    *voltage chasing currents*.  These currents give the stochastic spike emission process a component
    that adapts to the history of the voltage.  The equations used for the voltage chasing currents 
    are: 
    
    .. math:: 
        \frac{dQ_i}{dt} = r_i (V - Q_i)
        
    After the neuron spikes, the voltage chasing currents are reset to the value of the voltage 
    immediately following the spike: 
    
    .. math:: 
        Q_i \gets V_r
        
    The hazard rate is computed at each time step and compared to a uniformly distributed random number to 
    determine whether the neuron spikes here.  The computation of :math:`{\bf X}_t (t)` 
    at each time step is done by :meth:`update_X_arr`, while the inner product with the parameter vector 
    :math:`{\bf w}_t` and the random decision of whether a spike occurs or not is taken by :meth:`update`.
    """
    def __init__(self,t_bins,
                      volt_adapt_time_const=[],
                      dt=0.0001,
                      thresh_param=None,
                      thresh_param_dict={}):
        
        self.dt = dt
        self.t_bins = t_bins 
        self.bin_count = len(t_bins)
        #: max amount of time during which we care about previous spikes, see self.update
        self.t_max = t_bins[-1]
        self.t_hist = []
        self.X_arr = None 
        
        #: these are the time constants of the voltage chasing parameters
        self.volt_adapt_time_const = volt_adapt_time_const
        
        #: the actual value of the currents which will be an array
        self.volt_adapt_currents = None
        
        #: these parameters take the internal values and map them to a spike probability
        self.thresh_param = thresh_param
        self.thresh_param_dict = thresh_param_dict 
        
        #: current value of the spiking probability 
        self.spike_prob = None 
        
        self.param_ct = len(t_bins) + len(volt_adapt_time_const) + 2 
        
    def set_param(self,param):
        """
        After the optimization is done, this method allows the final 
        value of the parameter array to be set, and parses this parameter
        array into a parameter dictionary.  
        """
        self.thresh_param = param
        param_dict = {'full_thresh_param':list(self.thresh_param),
                      'v_param':self.thresh_param[0],
                      '1_param':self.thresh_param[1],
                      'thresh_shape_param':list(self.thresh_param[2:2+self.bin_count]),
                      'volt_adapt_param':list(self.thresh_param[2+self.bin_count:]),
                      'volt_adapt_time_const':self.volt_adapt_time_const,
                      "dt":self.dt,
                      't_bins':self.t_bins,
                      }
        
        self.param_dict = param_dict
    
    def _compute_ind_arr(self):
        """
        Computes array of integer-valued indicator functions that 
        only depend on the spike history.  The coefficients 
        of the paramater array will do an inner product with 
        this indicator array to specify the shape of the 
        'spike induced threshold'.  
        """
        
        ind_arr = np.zeros( (self.bin_count) )
        
        for (t_ind,t_bin) in enumerate(self.t_bins):
            ind_arr[t_ind] = len( [t for t in self.t_hist if t <= t_bin] )
            
        return ind_arr 
    
    def _compute_bin_number(self,t):
        """ 
        Which interval does t belong to? 
        """
        bin_number = 0
        if t < self.t_bins[0]:
            return bin_number 
        
        for t_cur in self.t_bins[1:]:
            bin_number += 1
            if t < t_cur:
                break
                
        return bin_number
            
    def reset(self,V=None):
        """
        This method resets to neuron to a resting state.  The spiking 
        history is erased and the the voltage adaption currents are either
        erased or are set to the current value of the voltage itself.
        
        :keyword V: value to which voltage chasing currents are reset to 
        """
        self.t_hist = [] 
        self.spike_prob = None 
        self.X_arr = np.nan 
        
        if (self.volt_adapt_time_const != None): 
            if V != None: 
                # if V is specified, we set all the adaptation currents to be exactly at V
                self.volt_adapt_currents = np.array([V for k in self.volt_adapt_time_const]) 
            else: 
                # just set to None, the _update_volt_dep_currents function will 
                #initialize the currents when the time comes 
                self.volt_adapt_currents = None 

    def _update_volt_dep_currents(self,V):
        """
        Updates voltage chasing currents.  These currents 
        helps the threshold to incorporate voltage dependence.  
        The degree to which these currents will contribute to 
        the actual spike probability is determined by the threshold 
        parameter array.
        """
        
        #: if we don't even have time constants for volt adapt, then forget about it
        if self.volt_adapt_time_const != None:
            # if the following value isn't set, we need to initialize it 
            if self.volt_adapt_currents == None: 
                self.volt_adapt_currents = np.array([V for k in self.volt_adapt_time_const])
            else: 
                for (ind,k) in enumerate(self.volt_adapt_time_const): 
                    old_volt_adapt_val = self.volt_adapt_currents[ind]
                    self.volt_adapt_currents[ind] =  old_volt_adapt_val + self.dt * k * (V - old_volt_adapt_val) 
        
            return self.volt_adapt_currents
        
    def update_X_arr(self,V):
        """
        Updates X_arr (the vector with which we do a dot product with 
        the threshold parameters to calculate the spiking probability).
        """
        
        # if the voltage is None, which means the neuron is spiking, 
        # don't worry about calculating a spike probability
        if V == None or np.isnan(V): 
            X_arr = np.array([np.nan] * self.param_ct) 
            self.X_arr = X_arr
            return X_arr 
        
        # we only update the spiking history if the neuron
        # is not currently spiking
        self._update_hist_by_dt()
        volt_adapt_arr = self._update_volt_dep_currents(V)
        ind_arr = self._compute_ind_arr()
        X_arr = np.concatenate(([V,1],ind_arr,volt_adapt_arr))
        self.X_arr = X_arr 
        
        return X_arr
    
    def _update_hist_by_dt(self):
        """
        Updates self.t_hist by dt.
        """
        if self.t_hist:
            self.t_hist = [t+self.dt for t in self.t_hist]
            
            if self.t_hist[0] >= self.t_max:
                self.t_hist.pop(0)
    
    def update(self,V):
        """
        Updates inner state and returns True if there is a spike, and False if 
        there is no spike. 
        """

        X_arr = self.update_X_arr(V)
        
        # if neuron is already spiking, then don't have the threshold 
        # spike again in its refractory period 
        if V == None or np.isnan(V): 
            self.spike_prob = 0
            return False  
        
        if self.thresh_param == None: 
            raise ValueError("Threshold parameter must be specified in self.thresh_param in order to compute spike probability!")
        
        hazard = exp(self.thresh_param.dot(X_arr)) 
        
        # TO DO: replace this approximation with:
        #spike_prob = 1 - exp(-1.0 * hazard)
        
        self.spike_prob = hazard
        
        if random.random() < self.spike_prob :
            self._spike()
            return True 
        else: 
            return False

    def _spike(self):
        # add a spike to spike history and set the voltage chasing currents to zero. 
        # once the neuron is done spiking the voltage chasing currents will be 
        # reset to the current voltage of the neuron, and then will be free to 
        # chase the neuron's current voltage. 
        
        self.t_hist.append(0.0)
        self.volt_adapt_currents = None

def max_likelihood(X_thresh_list,spike_ind_list,thresh_init,process_ct=None,iter_max=20,stopping_criteria=0.01):   
    """
    Performs maximum likelihood optimization.  
    """
    cur_param = thresh_init
    
    # we need to unpack some variables for better parallel consumption             
    X_arr_cat = None        
    prev_total_ind_ct = 0 
    
    for (ind,X_thresh) in enumerate(X_thresh_list):
        if X_arr_cat == None: 
            X_arr_cat = X_thresh
            spike_ind_arr_cat = spike_ind_list[ind]
        else:
            X_arr_cat = np.vstack( (X_arr_cat,X_thresh) )
            spike_ind_arr_cat = np.hstack( (spike_ind_arr_cat,spike_ind_list[ind] + prev_total_ind_ct) )
        
        prev_total_ind_ct += len(X_thresh)

    # how many workers are going to be used 
    if process_ct == None:
        process_ct = multiprocessing.cpu_count()
    
    #print "Process ct: " + str(process_ct)
    
    # splitting data into chuncks
    X_len = len(X_arr_cat)
    ind_vec = np.linspace(0, X_len, process_ct+1)
    ind_list = [int(this_ind) for this_ind in ind_vec[1:-1]]
    X_cat_split = np.split(X_arr_cat, ind_list,0)
    #print "Length of X_cat_split: " + str(len(X_cat_split))
    
    print "Starting max likelihood optimization..."
    
    for _ in range(iter_max):
        [grad,cur_param] = par_calc_log_like_update(cur_param, X_cat_split,process_ct)
        grad_norm = np.linalg.norm(grad)
        #print "Norm of log likelihood gradient: " + str(grad_norm)
        
        if grad_norm < stopping_criteria: 
            break
        
    return cur_param

def compute_log_likelihood(X_thresh_list,spike_ind_list,thresh_param):
    """
    Computes log likelihood of the spike trains given the parameters.  
    
    :param X_thresh_list: list of X_thresh matrices, one array for each sweep
    :param spike_ind_list: list of spike indices, one array for each sweep 
    :param thresh_param: threshold parameter array
    :returns: log likelihood of the time series given parameter array
    """
    
    log_sum = 0.0
    for (ind,X_thresh) in enumerate(X_thresh_list):
        
        spike_ind = spike_ind_list[ind]
        
        for (t,X_thresh_val) in enumerate(X_thresh):
            
            if np.isnan(X_thresh_val[0]):
                continue 
                
            if t+1 in spike_ind:
                log_sum += thresh_param.dot(X_thresh_val)
            else: 
                log_sum -= exp(thresh_param.dot(X_thresh_val)) 
                
    return log_sum 
     
def estimate_thresh_parameters(subthresh_obj,
                               thresh_obj,
                               raw_data,
                               process_ct=None,
                               max_lik_iter_max=25,
                               thresh_param_init=None,
                               stopping_criteria=0.01):
    r"""
    Estimates threshold parameters that fit the raw data, and the particular 
    form of the threshold and subthreshold components of the model. 
    
    :param subthreshold_obj: :class:`fit_neuron.optimize.subthreshold.Voltage` instance
    :param thresh_obj: :class:`fit_neuron.optimize.threshold.StochasticThresh` instance
    :param raw_data: :class:`fit_neuron.data.my_data.RawData` instance 
    :param make_lik_iter_max: maximum number of max likelihood iterations
    :param process_ct: number of processors we want to use to distribute max likelihood; if None then use CPU count.
    :param stopping_criteria: minimum :math:`L^2` norm of gradient of log likelihood gradient below which we stop optimization.
    :returns: array of threshold parameters
    """
        
    print "Estimating threshold parameters..."
    param_ct = thresh_obj.param_ct
    
    if thresh_param_init == None:     
        thresh_param_init = np.ones( (param_ct) ) * 0.001
    
    X_thresh_list = []
    spike_ind_list = data.extract_spikes.spk_from_bio(raw_data.membrane_voltage_list)
    
    for (ind,sweep) in enumerate(raw_data):
        
        subthresh_obj.reset()
        thresh_obj.reset()
        arr_len = len(sweep.membrane_voltage)
       
        spike_ind = spike_ind_list[ind]
        
        V = sweep.membrane_voltage[0]
        X_thresh = np.zeros( (arr_len,param_ct) )
        
        for (t,Ie) in enumerate(sweep.input_current):
            
            V = subthresh_obj.update(V,Ie)
            new_X = thresh_obj.update_X_arr(V)
            X_thresh[t,:] = new_X
            
            # force the model to spike when the biological neuron spikes
            if t+1 in spike_ind:
                subthresh_obj.spike()
                thresh_obj._spike()
        
        X_thresh_list.append(X_thresh)
    
    thresh_param = max_likelihood(X_thresh_list,
                                  spike_ind_list,
                                  thresh_param_init,
                                  process_ct=process_ct,
                                  iter_max=max_lik_iter_max,
                                  stopping_criteria=stopping_criteria)
    
    log_l = compute_log_likelihood(X_thresh_list,spike_ind_list,thresh_param)
    print "Log likelihood: " + str(log_l)
    return thresh_param
                    

