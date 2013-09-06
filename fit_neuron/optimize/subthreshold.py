""" 
subthreshold
~~~~~~~~~~~~~
Parameter extraction of subthreshold parameters from voltage clamp data.   
This module defines a subthreshold model (:class:`Voltage`) 
and defines an optimization function (:func:`estimate_volt_parameters`). 
"""

import numpy as np
    
class Voltage():
    r"""
    This class defines the voltage predictions of the gLIF models.
        
    :keyword param_dict: dictionary of parameter values.
    :keyword param_arr: array of parameter values.
    :keyword sic_list: list of :class:`fit_neuron.optimize.sic_lib.SicBase` instances.
    :keyword volt_nonlin_fcn: voltage nonlinearity function.
    :keyword Vr: reset value of the voltage following a spike and a refractory period.
    :keyword t_ref: length of refractory period in seconds.
    :keyword dt: time increment.
        
    The difference equation describing the evolution of the membrane potential
    is the following:
    
    .. math::
        V(t + dt) - V(t) = \alpha_1 + \alpha_p V + \alpha_{I_e} I_e + \alpha_g g(V) + \sum_{i=1}^{i=n} \beta_i I_i(t,\{\hat{t}\},V)

    where :math:`\{\hat{t}\}` are the previous spike times relative to the current time :math:`t`, 
    the :math:`I_i` are the spike induced currents, which may or may not depend explicitly on :math:`V`,
    and :math:`g(V)` is an optional voltage nonlinearity function.
    
    The updates are implemented as follows: 
    
    .. math:: 
        V(t + dt) - V(t) = {\bf w^{\top} X_{sub}}(t)
        
    where :math:`\bf w` is the parameter vector, and 
    
    .. math::
        {\bf X_{sub}} = [V,1,I_e,g(V),{\bf I}(t)^{\top}]^{\top}
        
    The vector :math:`\bf X_{sub}` is computed at each time step by 
    :meth:`compute_X_subthresh` and the inner product is taken 
    by :meth:`update`.
    """
    
    def __init__(self,param_dict={},
                 param_arr=[],
                 sic_list=[],
                 volt_nonlin_fcn=None,
                 dt=0.0001,
                 Vr=-70,
                 t_ref=0.004):
        
        [sic.reset() for sic in sic_list]
        self.sic_list = sic_list 
        
        #: dictionary of parameter values
        self.param_dict = param_dict 
        
        #: array of subthreshold parameter values
        self.param_arr = param_arr
        
        #: voltage nonlinearity function, may be None
        self.volt_nonlin_fcn = volt_nonlin_fcn
        
        #: time step
        self.dt = dt
        
        #: reset potential
        self.Vr = Vr
                
        #: refractory period after each spike during which the neuron will
        #: have a value of numpy.nan
        self.t_ref = t_ref
        
        #: A counter used to count down the time during which the neuron spikes.
        #: Whenever a spike occurs, its value is set to :attr:`t_ref`.
        self.t_spike_counter = 0
        
        #: values of the spike induced currents
        self.SIC_values = []
        
        #: current spiking state of the neuron
        self.is_spiking = False
        
        #: how many parameters does the neuron need?
        param_ct = 3 + len(sic_list)
        
        if volt_nonlin_fcn != None:
            param_ct += 1
            
        #: number of parameters and hence length of :attr:`param_arr`.
        self.param_ct = param_ct
                
    def reset(self):
        """
        Resets all the spike induced currents to resting state 
        by calling the reset method for all the spike 
        induced currents (eg. see :meth:`fit_neuron.optimize.sic_lib.ExpDecay_sic.reset`).
        """
        [sic.reset() for sic in self.sic_list]
        self.is_spiking = False
        
    def spike(self):
        """
        Calls the :meth:`spike` method of all spike induced 
        current, sets :attr:`is_spiking` to True, and sets the 
        timer of the spike to :attr:`t_ref`.  
        """
        [sic.spike() for sic in self.sic_list]
        self.is_spiking = True 
        self.t_spike_counter = self.t_ref
        
        return None 
    
    def update(self,V=None,Ie=None):
        """
        This method takes a voltage value and an input current value and returns
        the value of the voltage at the next time step.  If the neuron is 
        not currently is a spiking state, the method will return a float. 
        If the neuron is in a spiking state, the method will return a :class:`numpy.nan`. 
        value.
        """
        
        if self.is_spiking:
            V_new = self.update_spike_counter()
            return V_new
            
        X_subthresh = self.compute_X_subthresh(V,Ie)
        V_new = V + self.param_arr.dot(X_subthresh)
        return V_new 
        
    def update_spike_counter(self):
        """
        This method, only to be called when the neuron is currently 
        spiking, updates the counter of the spike.  Once the neuron 
        has been in a spiking state for a period of time longer 
        than :attr:`t_ref`, the neuron will exit the spiking state 
        by setting :attr:`is_spiking` back to True.
        """
        self.t_spike_counter -= self.dt
    
        if self.t_spike_counter <= 0:
            self.is_spiking = False 
            self.t_spike_counter = 0
            V_new = self.Vr
        else:
            V_new = np.nan  
            
        return V_new 
    
    def compute_X_subthresh(self,V,Ie):
        r"""
        Updates the values of the spike induced currents and 
        then computes :math:`\bf X_{sub}`.  As explained in the class 
        docstring, the voltage difference is computed as the inner product 
        of  :math:`\bf X_{sub}` with the parameter vector :math:`\bf w`.  
        
        .. note:: 
            This method can be called even if :attr:`param_arr` has not yet been computed.
            In fact, the function :func:`setup_regression` uses this functionality
            to compute  :math:`\bf X_{sub}` at every time point so that linear regression can then 
            be easily performed.  This practice ensures that the regressed parameters match 
            the :meth:`compute_X_subthresh` method without any indexing mismatches.  It also 
            encourages code reuse.   
        """
        
        [sic.update(V) for sic in self.sic_list]
        spike_currents = [sic.sic_val for sic in self.sic_list]
        
        if self.volt_nonlin_fcn:
            cur_volt_nonlin = [self.volt_nonlin_fcn(V)]
        else:
            cur_volt_nonlin = []
        
        X_subthresh = np.concatenate(([V,1,Ie],cur_volt_nonlin,spike_currents))
        return X_subthresh
    
    def set_param(self,param_arr):
        """
        Method to be called after :func:`estimate_volt_parameters` has 
        found an optimal set of parameters.  The method saves the input
        arrays as its :attr:`param_arr` attribute and parses this array
        as a dictionary and saves it as the instance's 
        :attr:`param_dict` attribute.  
        
        :param param_arr: array of subthreshold parameter values.
        """
        param_dict = {'full_param_arr':list(param_arr),
                      'v_param':param_arr[0],
                      '1_param':param_arr[1],
                      'Ie_param':param_arr[2]}
        
        if self.volt_nonlin_fcn != None: 
            param_dict.update({'nonlin_param':param_arr[3]})
            param_dict.update({'sic_param_list':list(param_arr[4:])})
        else: 
            param_dict.update({'sic_param_list':list(param_arr[3:])})
            
        self.param_arr = param_arr
        self.param_dict = param_dict
    
def setup_regression(subthresh_obj,sweep):
    
    # first initialize spike induced currents 
    subthresh_obj.reset()

    # pre-allocate memory
    arr_len = len(sweep.input_current)
    X = np.zeros( (arr_len,subthresh_obj.param_ct) )
    Y = np.zeros( (arr_len) )

    row_ct = 0

    for ind in range(0,arr_len-1):
        
        if ind+1 in sweep.reset_ind:
            subthresh_obj.spike()
            continue
      
        # voltage difference we are trying to fit
        Y[row_ct] = sweep.membrane_voltage[ind+1] - sweep.membrane_voltage[ind]
        
        # regressor we want to use to fit to voltage difference
        V = sweep.membrane_voltage[ind]
        Ie = sweep.input_current[ind]
        X_subthresh = subthresh_obj.compute_X_subthresh(V,Ie)
        X[row_ct] = X_subthresh
                
        row_ct += 1 
    
    ind_remove = range(row_ct,arr_len)
    Y = np.delete(Y,ind_remove)
    X = np.delete(X,ind_remove,0)

    return [X,Y]
    
def estimate_volt_parameters(subthresh_obj,processed_data):
    r"""
    Estimates voltage parameters using data provided and stores these as attributes. 
    Does a least squares linear regression of the voltage parameters. 
    
    :param subthresh_obj: subthreshold part of model. 
    :type subthresh_obj: :class:`Voltage`
    :param processed_data: data with the spikes removed. 
    :type processed_data: :class:`fit_neuron.data.my_data.ProcessedData`
    :returns: array of subthreshold parameters.
    
    The equation we are solving is the following: 
    
    .. math:: 
        \min_{b} \|Xb - Y\|^2
        
    where 
    
    .. math:: 
        X = \begin{bmatrix}
        V(0) & 1 & I_e(0) & g(V) & I_0(0) & \hdots & I_n(0)  \\
        V(1) & 1 & I_e(1) & g(V) & I_0(1) & \hdots & I_n(1) \\
        \vdots & \vdots & \vdots & \vdots &\vdots  & \vdots  & \vdots 
        \end{bmatrix}
    
    and 
    
    .. math:: 
        Y = \begin{bmatrix}
        V(1) - V(0) \\ 
        V(2) - V(1) \\
        \vdots
        \end{bmatrix}
        
    The value of :math:`b` that minimizes this expression is the parameter
    vector for the subthreshold object. 
    """
    
    print "Estimating subthreshold parameters..."
    
    X_sum = None 
    Y_sum = None 
    
    for sweep in processed_data:
        [X,Y] = setup_regression(subthresh_obj,sweep)

        if not X_sum == None: 
            X_sum = np.vstack( (X_sum,X) )
            Y_sum = np.hstack( (Y_sum,Y) )
        else: 
            X_sum = X 
            Y_sum = Y

    param_arr  = np.linalg.lstsq(X,Y)[0]

    return param_arr 
    