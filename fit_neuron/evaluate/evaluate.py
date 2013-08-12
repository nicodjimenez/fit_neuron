""" 
This simulation file runs the subthreshold model together with 
the threshold model and plots the results against the 
actual recordings. 
"""
import os
import pylab 
import math
import numpy as np
import spkd_lib
from fit_neuron.data import spk_from_bio, spk_from_sim

def simulate(neuron=None,input_current_list=[],membrane_voltage_list=[],file_id_list=[],reps=1):
    """
    [simulated_voltage_list,sim_file_id_list] = evaluate.simulate(neuron,input_current_list,file_id_list)

    This function takes a neuron fitted from some data and 
    simulates it against the data it is supposed to fit.  
    
    :param neuron: a neuron instance
    :param input_current_list: list of input currents
    :param membrane_voltage_list: list of voltage trace, useful for initializing neuron
    :param file_id_list: list of file ID's corresponding to input_current_list
    :param reps: number of times we want to produce simulated voltage traces for each input current in input current list
    :returns: dictionary of {file_id: simulated_voltage_list}
    
    .. note:: 
        The values in the dictionary will be lists that have lengths corresponding
        to reps.  Each element in these lists will be an array, 
        computed as a single monte carlo trial.
    """
    
    simulated_voltage_dict = {}
    
    for (ind,input_current) in enumerate(input_current_list): 
        
        cur_file_id = file_id_list[ind]
        V_init = membrane_voltage_list[ind][0]
        cur_voltage_list = []
        #print "Evaluating input current number: " + str(ind)
        
        for cur_rep in range(reps):
            #print "Current repetition: " + str(cur_rep)
            
            arr_len = len(input_current)
            simulated_voltage_arr = np.zeros((arr_len))
            simulated_voltage_arr[0] = V_init
            neuron.reset(V_init)
            
            t = 0
            while t < arr_len - 1:
                simulated_voltage_arr[t+1] = neuron.update(input_current[t])
                t += 1
            
            cur_voltage_list.append(simulated_voltage_arr)
            #simulated_voltage_dict.update(cur_file:)
        
        simulated_voltage_dict.update({cur_file_id:cur_voltage_list})
                        
    return simulated_voltage_dict

def plot_sim_vs_real(simulated_voltage_dict=[],
                     bio_voltage_dict=[],
                     input_current_dict=[],
                     plot_nan_as_spike=True,
                     fig_dir="./",
                     file_ext=".png"):
    """
    Plots the simulated voltage traces against the actual voltage traces and 
    saves the result as .eps file.  
    
    :param bio_voltage_dict: dictionary, values of which are arrays of voltage traces
    :param input_voltage_dict: dictionary of input current injections, values of which are arrays of input stimuli
    :param simulated_voltage_dict: dictionary. values of which are lists of arrays of monte carlo simulated voltage traces
    :param fig_dir: directory into which figures will be plotted
    :keyword nan_as_spike: do we draw spiking lines where the traces are nan?
    :type nan_as_spike: bool 
    :keyword file_ext: file extension of plotting results
    :type file_ext: str
    
    .. note:: 
        It is assumed that the keys of the dictionary inputs correspond to each 
        other.  
    """
    
    for key in input_current_dict.keys():
    
        input_current = input_current_dict[key]
        membrane_voltage = bio_voltage_dict[key]
        membrane_voltage_sim_list = simulated_voltage_dict[key]
        
        for (cur_rep,membrane_voltage_sim) in enumerate(membrane_voltage_sim_list):
                
            pylab.figure(figsize=(20,12), dpi=200)
            data_len = len(membrane_voltage)
            x = np.array(range(0,data_len))
            y1 = membrane_voltage
            y2 = membrane_voltage_sim
            y3 = input_current
            
            subplot_top = pylab.subplot(211)
            pylab.title(key) 
            pylab.plot(x, y1, color="blue", linewidth=1.0, linestyle="-", label="Membrane voltage")
            pylab.plot(x, y2, color="green", linewidth=1.0, linestyle="-", label="Simulated voltage")
            subplot_top.legend()
            
            if plot_nan_as_spike:
                spike_ind_sim = spk_from_sim([membrane_voltage_sim])[0]
            else:
                spike_ind_sim = []
                
            [pylab.axvline(x=cur_ind,linestyle = ':',color='green') for cur_ind in spike_ind_sim]
            pylab.ylim( (-80,-20) )
            pylab.xlim( (0,len(x)) )
            ticks = np.arange(x.min(), x.max(), 1000)
            labels = 0.1*np.array(range(ticks.size))
            pylab.xticks(ticks,labels)
            pylab.xlabel('Time (s)')
            pylab.ylabel('Voltage (mV)')
            
            #gamma = spkd_lib.gamma_factor(spike_ind_sim,spike_ind_raw,0.005)
            #gamma_str = r"$\Gamma =" + str(gamma) + "$" 
            
            subplot_low = pylab.subplot(212)
            pylab.plot(x, y3, color="black", linewidth=1.0, linestyle="-",label="Input current")
            subplot_low.legend()
            pylab.xlim( (0,len(x)) )
            ticks = np.arange(x.min(), x.max(), 1000)
            labels = 0.1*np.array(range(ticks.size))
            pylab.xticks(ticks,labels)
            pylab.xlabel('Time (s)')
            pylab.ylabel('Current (A)')
            
            file_str = os.path.splitext(key)[0] + "_rep" + str(cur_rep) + file_ext
            file_path = os.path.join(fig_dir,file_str)     
            pylab.savefig(file_path)   
            pylab.close()
            print "New plot: " + file_path
        
def plot_spk_performance_metrics(bio_voltage_dict,simulated_voltage_dict,fig_dir="./",file_ext=".png",dt=0.0001):
    """
    Plots spike distance metrics for the data provided.  Metrics used are the gamma coincidence factor, 
    the Schrieber similarity, and the van Rossum distance.  Each is plotted as a function of the
    smoothness parameter for that particular metric.  The spike times are extracted 
    from the voltage traces using :func:`fit_neuron.data.extract_spikes.spk_from_sim` 
    and :func:`fit_neuron.data.extract_spikes.spk_from_bio`. 
    
    :param bio_voltage_dict: recorded voltage traces, indexed by a file ID
    :type bio_voltage_dict: dict
    :param simulated_voltage_dict: simulated voltage traces, indexed by a file ID
    :type simulated_voltage_dict: dict 
    :keyword fig_dir: directory where the plots will be saved 
    :keyword file_ext: file extension 
    :keyword dt: time step interval 
    
    .. note::
        The keys of bio_voltage_dict and simulated_voltage_dict must match.  
        In addition, there may be multiple simulations for a single file ID, 
        but there should be a single recorded voltage trace for a file ID.
    """
    
    for key in bio_voltage_dict.keys():
        membrane_voltage = bio_voltage_dict[key]
        membrane_voltage_sim_list = list(simulated_voltage_dict[key])
        
        for (cur_rep,membrane_voltage_sim) in enumerate(membrane_voltage_sim_list):
            key_id = os.path.splitext(key)[0]
            end_ext = "_" + key_id + "_rep" + str(cur_rep) + file_ext 
            sim_spk_times = dt * spk_from_sim([membrane_voltage_sim])[0]
            bio_spk_times = dt * spk_from_bio([membrane_voltage])[0]
            
            # --------- VAN ROSSUM -------------
            powers = np.arange(-5,5,0.25)
            van_rossum_i = [math.pow(10,power) for power in powers]
            # as tc increases, the spikes get smoothed more and more so distance will decrease
            van_rossum_d = [spkd_lib.van_rossum_dist(bio_spk_times,sim_spk_times,tc=t_cur) for t_cur in van_rossum_i] 
            pylab.semilogx(van_rossum_i, van_rossum_d)
            file_str = "van_rossum_dist" + end_ext
            file_path = os.path.join(fig_dir,file_str)
            pylab.grid()
            pylab.xlabel(r'$\tau_r$')
            pylab.ylabel('van Rossum distance')
            pylab.savefig(file_path)  
            print "New plot: " + file_path
            pylab.close()
            
            # --------- GAMMA FACTOR ----------------
            gamma_factor_i = np.arange(0.0,0.01,0.00005)
            gamma_factor_d = [spkd_lib.gamma_factor(sim_spk_times,bio_spk_times,delta=cur_delta) for cur_delta in gamma_factor_i]
            pylab.plot(gamma_factor_i,gamma_factor_d)
            pylab.xlabel(r'$\Delta t$')
            pylab.ylabel('Gamma coincidence factor')
            
            if max(gamma_factor_d) > 0:
                pylab.ylim( (-0.01,1.01) )
                
            pylab.grid()
            file_str = "gamma_factor" + end_ext
            file_path = os.path.join(fig_dir,file_str)
            pylab.savefig(file_path)  
            print "New plot: " + file_path
            pylab.close()
            
            # ---------- SCHRIEBER SIMILARITY ----------
            powers = np.arange(-5,5,0.25)
            schrieber_i = [math.pow(10,power) for power in powers]
            schrieber_d = [spkd_lib.schrieber_sim(sim_spk_times,bio_spk_times,sigma=cur_sigma) for cur_sigma in schrieber_i]
            pylab.semilogx(schrieber_i,schrieber_d)
            pylab.xlabel(r'$\sigma$ (bandwidth of Gaussian kernel)')
            pylab.ylabel('Schrieber similarity')
            
            if max(gamma_factor_d) > 0:
                pylab.ylim( (-0.01,1.01) )
            
            pylab.grid()
            file_str = "schrieber_similarity" + end_ext
            file_path = os.path.join(fig_dir,file_str)
            pylab.savefig(file_path)  
            print "New plot: " + file_path
            pylab.close()

            '''
            # --------- VICTOR PURPURA -------------
            victor_purpura_i = np.arange(0.0,10,0.1)
            victor_purpura_d = [spkd_lib.victor_purpura_dist(bio_spk_times,sim_spk_times,cost=cur_cost) for cur_cost in victor_purpura_i]
            pylab.plot(victor_purpura_i,victor_purpura_d)
            file_str = "victor_purpura" + end_ext
            file_path = os.path.join(fig_dir,file_str)
            pylab.savefig(file_path)  
            print "New plot: " + file_path
            pylab.close()
            '''
