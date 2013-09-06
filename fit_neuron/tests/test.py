import os
import pickle
import json
from fit_neuron import evaluate
from fit_neuron.data import load_neuron_data
import fit_neuron.optimize
from fit_neuron.optimize import sic_lib

T_BIN_DEFAULT = [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0008,0.001,0.00125,0.0015,0.002,0.003,0.004,0.005,0.01,0.015,0.02,0.025,0.03,0.05,0.08,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.7,0.9,1.2]

# where the outputs of optimization (figures, json files,...) will be saved
OUTPUT_DIR = "test_output_figures"

def test_fit_neuron(output_dir=OUTPUT_DIR,neuron_num = 1):
    """
    Example procedure for estimating a LIF model for a single neuron.  
    
    :param output_dir: directory where JSON files and pickle files summarizing fitting results will be saved.
    :param neuron_num: number of neuron we want to fit (number ranges from 1 to 12)
    
    .. note:: 
        The output JSON files will contain a dictionary of parameter values.  
        The output pickle file will contain a :mod:`fit_neuron.optimize.neuron_base_obj.Neuron` instance 
        which can simulate the artificial neuron.
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
        
    # -------------- LOADING DATA  -----------------
    
    # if my_input_type is set to "noise_only" then only load noisy inputs
    (file_id_list,input_current_list,membrane_voltage_list,dt) = load_neuron_data(neuron_num,
                                                                                  input_type="noise_only",
                                                                                  max_file_ct=4) 
    # -------------- RUNNING OPTIMIZATION  -----------------
    
    t_bins = T_BIN_DEFAULT
    sic_list = [sic_lib.StepSic(t,dt=dt) for t in t_bins]
           
    neuron = fit_neuron.optimize.fit_neuron(input_current_list=input_current_list,
                                     membrane_voltage_list=membrane_voltage_list,
                                     dt=dt,
                                     process_ct=None,
                                     max_lik_iter_max=25,
                                     stopping_criteria=0.01,
                                     sic_list=sic_list)

    # -------------- SAVING RESULTS  -----------------
    
    folder_id = "neuron_" + str(neuron_num)
    neuron_folder = os.path.join(output_dir,folder_id)
    
    if not os.path.exists(neuron_folder):
        os.makedirs(neuron_folder)
    
    pickle_file_dir =  os.path.join(neuron_folder,"pickle_files")
    
    if not os.path.exists(pickle_file_dir):
        os.makedirs(pickle_file_dir)
    
    pickle_file_path = os.path.join(pickle_file_dir,folder_id + ".p")
    print "Successfully pickle file to: " + str(pickle_file_path)
    
    pickle.dump(neuron, open(pickle_file_path,'wb'))
    
    json_file_dir = os.path.join(neuron_folder,"json_files")
    
    if not os.path.exists(json_file_dir):
        os.makedirs(json_file_dir)
    
    json_file_path = os.path.join(json_file_dir,folder_id + ".json")
    json.dump(neuron.get_param_dict(),open(json_file_path,'wb'),sort_keys = False, indent = 4)
    
    print "Successfully saved json file to: " + str(json_file_path)

def test_eval_sav_results(output_dir=OUTPUT_DIR,neuron_num=1):
    """
    Example procedure for evaluating the estimated model parameters 
    gerenated by :func:`test_fit_neuron`.
    
    :param output_dir: directory where JSON files and pickle files summarizing fitting results have be saved.
    :param neuron_num: number of neuron we want to evaluate (number ranges from 1 to 12)
    
    .. warning:: 
        This function will not work unless :func:`test_fit_neuron` has 
        already been run before.  
    """

    # ------------ LOADING SAVED RESULTS -----------------
    folder_id = "neuron_" + str(neuron_num)
    neuron_folder = os.path.join(output_dir,folder_id)
    
    if not os.path.exists(neuron_folder):
        raise ValueError("Neuron data cannot be found in " + str(neuron_folder))
    
    pickle_file_dir =  os.path.join(neuron_folder,"pickle_files")
    pickle_file_path = os.path.join(pickle_file_dir,folder_id + ".p")
    neuron = pickle.load(open(pickle_file_path,'rb'))

    (file_id_list,input_current_list,membrane_voltage_list,dt) = load_neuron_data(neuron_num,
                                                                                  input_type="noise_only",
                                                                                  max_file_ct=4) 
        
    # -------------- PLOTTING OUTPUT FIGURES --------------------

    print "Running monte carlo simulations..."
    simulated_voltage_dict = evaluate.simulate(neuron=neuron,
                                              input_current_list=input_current_list,
                                              membrane_voltage_list=membrane_voltage_list,
                                              file_id_list=file_id_list,
                                              reps=10)
            
    bio_voltage_dict = dict(zip(file_id_list,membrane_voltage_list))
    input_current_dict = dict(zip(file_id_list,input_current_list))

    figure_dir = os.path.join(neuron_folder,"figures")
    
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    # plots simulation results in a specified directory
    evaluate.plot_sim_vs_real(simulated_voltage_dict=simulated_voltage_dict,
                              bio_voltage_dict=bio_voltage_dict,
                              input_current_dict=input_current_dict,
                              fig_dir=figure_dir)
    
    stats_dir = os.path.join(neuron_folder,"stats")
    
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    
    evaluate.plot_spk_performance_metrics(bio_voltage_dict,simulated_voltage_dict,fig_dir=stats_dir)
    
if __name__ == '__main__':
    test_fit_neuron()
    test_eval_sav_results()

