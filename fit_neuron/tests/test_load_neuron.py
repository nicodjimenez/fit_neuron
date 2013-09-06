"""
Tests the functionality of the models themselves, independent of 
fitting procedures.  
"""

from fit_neuron.optimize import Voltage, StochasticThresh, sic_lib, Neuron
from numpy import array,zeros, arange
import pylab 

def test_load_neuron():
    """
    Loads a neuron instance directly from parameter arrays and simulate 
    a step input current.  
    """
    t_bins = [0.0001,0.001,0.01,0.1]
    dt = 0.0001
    sic_list = [sic_lib.StepSic(t,dt=dt) for t in t_bins]
    param_arr = array([-0.015, -1.0, 1000000000.0,-0.01,0.005,-0.2,-0.1])
    thresh_param = array([1.2,56.0,-0.08,-0.05,-0.05,-0.04])
    subthresh_obj = Voltage(param_arr=param_arr, sic_list=sic_list, dt=dt, Vr=-70, t_ref=0.004)
    thresh_obj = StochasticThresh(t_bins=t_bins,dt=dt,thresh_param=thresh_param)
    neuron = Neuron(subthresh_obj=subthresh_obj,thresh_obj=thresh_obj,V_init=-70)
    
    v_arr = zeros( (500) )
    
    for ind in range(500):
        V_new = neuron.update(1E-10)
        v_arr[ind] = V_new 
        
    assert ( round(v_arr[-1]) - round(v_arr[0]) ) == 10.0

def plot_test_load_neuron():
    """
    Loads a neuron instance directly from parameter arrays and simulate 
    a step input current.  
    """
    t_bins = [0.0001,0.001,0.01,0.1]
    dt = 0.0001
    sic_list = [sic_lib.StepSic(t,dt=dt) for t in t_bins]
    param_arr = array([-0.015, -1.0, 1000000000.0,-0.01,0.005,-0.2,-0.1])
    thresh_param = array([1.2,56.0,-0.08,-0.05,-0.05,-0.04])
    subthresh_obj = Voltage(param_arr=param_arr, sic_list=sic_list, dt=dt, Vr=-70, t_ref=0.004)
    thresh_obj = StochasticThresh(t_bins=t_bins,dt=dt,thresh_param=thresh_param)
    neuron = Neuron(subthresh_obj=subthresh_obj,thresh_obj=thresh_obj,V_init=-70)
    
    v_arr = zeros( (500) )
    
    for ind in range(500):
        V_new = neuron.update(1E-10)
        v_arr[ind] = V_new 
    
    t_arr = dt * arange(500)
    pylab.plot(t_arr,v_arr)
    pylab.xlabel("Time (s)")
    pylab.ylabel("Voltage (mV)")
    pylab.title("Response to Test Current")
    pylab.show() 
    
if __name__ == '__main__':
    plot_test_load_neuron()
    
