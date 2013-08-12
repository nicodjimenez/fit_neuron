import numpy as np

class Neuron():
    """
    Return type of optimization procedure.  
    """
    def __init__(self,subthresh_obj=None,thresh_obj=None,V_init=None):
        self._subthresh_obj = subthresh_obj
        self._thresh_obj = thresh_obj
        self.V = V_init
        
    def reset(self,V_init=None,**kwargs):
        """
        Sets the neuron to a *resting state* with a specified voltage value.
        Resting state means that all spike induced currents are set to zero
        and the spiking memory of the neuron is erased.  
        
        :param V_init: initial value of voltage
        """
        self.V = V_init
        self._subthresh_obj.reset()
        self._thresh_obj.reset()
    
    def update(self,Ie):
        """
        Updates state of the neuron according to value 
        of external current by time delta of dt
        (specified in threshold, subthreshold objects).
        
        :param Ie: value of the external current  
        :returns: value of the new voltage.  If neuron is spiking, returns nan.
        """
        
        is_spike = self._thresh_obj.update(self.V)
        
        if is_spike == True: 
            self._subthresh_obj.spike()
        
        self.V = self._subthresh_obj.update(Ie=Ie,V=self.V)
        return self.V
        
    def get_param_dict(self):
        """
        Extracts and returns parameter dictionaries from the subthreshold 
        and threshold parts of the neuron. 
        """
        total_param_dict = {}
        total_param_dict.update({"subthreshold_parameters" :self._subthresh_obj.param_dict})     
        total_param_dict.update({"threshold parameters": self._thresh_obj.param_dict})
        return total_param_dict
        
if __name__ == '__main__':
    pass
    
