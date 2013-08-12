"""
This file will contain a library of spike induced current (sic) objects.
A user may provide any list of spike induced current objects to the optimization functions 
as long as all of these objects are derived from the :class:`SicBase` abstract class or 
implement methods with the same names and input arguments.
"""

import sys
import os 
from numpy import exp, arange

class SicBase():
    def __init__(self,**kwargs):
        raise NotImplementedError("Subclass must implement abstract method") 
    
    def update(self,V):
        r"""
        Updates value of current by timestep :math:`dt`. 
        
        :rtype: float
        :returns: new value of the spike induced current
        """
        raise NotImplementedError("Subclass must implement abstract method")  
    
    def spike(self):
        """
        Updates state of the current whenever the neuron spikes. 
        
        :rtype: None
        """
        raise NotImplementedError("Subclass must implement abstract method")  
    
    def reset(self):
        """
        Sets the value of the spike induced current to zero.  
        
        :rtype: None
        """
        raise NotImplementedError("Subclass must implement abstract method")  
    
class StepSic(SicBase):
    r"""
    Step wise spike induced current that is the sum of indicator 
    variables for the spiking history of the time since the last spike 
    being between zero and some t_max.  
    """
    def __init__(self,t_max,dt=0.0001):
        
        #: time defining indicator functions
        self.t_max = t_max
        
        #: actual value of the spike induced current
        self.sic_val = 0.0
    
        #: list of time elapsed since last spikes 
        self.t_hist = []
        
        #: time increments
        self.dt = dt
        
    def update(self,V):
        self.t_hist = [t + self.dt for t in self.t_hist if t <= self.t_max] 
        self.sic_val = len(self.t_hist)
        return self.sic_val
    
    def spike(self):
        self.t_hist.append(0)  
        self.sic_val = len(self.t_hist)

    def reset(self):
        self.sic_val = 0 
        self.t_hist = []

class ExpDecay_sic(SicBase):
    r"""
    Exponentially decaying spike induced current.  The class models
    the following differential equation:  
    
    .. math::
        \frac{dI}{dt} = -kI  
        
    When the neuron spikes, the current :math:`I` is incremented as follows: 
    
    .. math:: 
        I \gets I + 1 
    """
    
    def __init__(self,k=None,dt=0.0001):

        #: value of the spike induced current
        self.sic_val = 0.0 
        
        #: time step 
        self.dt = dt 
        
        #: decay rate 
        self.k = k 
        self.decay_factor = exp(-dt*k)
    
    def update(self,V):
        """
        Updates :attr:`sic_val` by applying exponential decay 
        by a time step :attr:`dt`.
        """
        self.sic_val = self.sic_val * self.decay_factor
        return self.sic_val
    
    def spike(self):
        """
        Additive rule called whenever the neuron spikes.  
        The value of :attr:`sic_val` is incremented by 1.  
        """
        self.sic_val += 1 
        return self.sic_val
        
    def reset(self):
        """
        Sets :attr:`sic_val` to zero.  
        This sets the spike induced current to a resting state.
        """
        self.sic_val = 0 

if __name__ == '__main__':
    sic_list = [ExpDecay_sic(k) for k in [5,10,15]]
    [sic.spike() for sic in sic_list]
    [sic.update(V=None) for sic in sic_list]
    for sic in sic_list: 
        print sic.sic_val 
