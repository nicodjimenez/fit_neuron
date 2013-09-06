"""
Allows the user to easily load sample patch clamp recordings
that were previously downloaded by :mod:`fit_neuron.data.dl_neuron_data`. 

In addition, the user may use :func:`is_noise` to filter current injections 
that are above an auto-correlation cutoff. 

.. warning:: 
    The functions :func:`load_neuron_files` and :func:`load_neuron_data` not work 
    unless one of the following is true: 
    
    #. The user has downloaded and unzipped the data via the following 
       script:
    
        .. code-block:: none
        
            sudo python -m fit_neuron.data.dl_neuron_data

            
    #.  The user manually downloaded and unzipped DataTextFiles.zip 
        (currently available at https://xp-dev.com/svn/neuro_fit/fit_neuron/data/DataTextFiles.zip)
        and the user specifies the path of the DataTextFiles directory as keyword arguments to 
        :func:`load_neuron_files` or :func:`load_neuron_data`.
        
""" 

import os
import warnings 
import numpy as np
from numpy import loadtxt
from scipy import stats

# current directory of this file
CUR_DIR = os.path.abspath(os.path.dirname(__file__))

# directory where data data should be unzipped
DATA_DIR = os.path.join(CUR_DIR,"DataTextFiles")

# url containing all the data
ZIP_ADDRESS = "https://xp-dev.com/svn/neuro_fit/fit_neuron/data/DataTextFiles.zip"

def is_noise(input_current,correlation_cutoff=0.8):
    """
    Given an array of equally spaced input current values, is the input noise or not?
    Use the Pearson auto-correlation coefficient of the input signal to determine this heuristically.
    
    :param correlation_cutoff: threshold below which we classify an input current as being noise
    :param input_current: a numpy array of equally spaced values 
    :keyword correlation_cutoff: threshold below which we classify an input current as being noise
    :returns: True if input_current is noise, False if not 
    
    >>> [file_id_list,input_list,voltage_list,time_delta] = load_neuron_files(neuron_num=1,file_num_list=range(1,16),show_print=False)
    >>> is_noise(input_list[0])
    False
    >>> is_noise(input_list[13])
    True
    """
    
    IND_DELAY = 200
    input_corr = stats.pearsonr( input_current[IND_DELAY:] , input_current[:-IND_DELAY] )
    
    #print "Inpur corr: " + str(input_corr[0])

    if input_corr[0] < correlation_cutoff: 
        return True 
    else: 
        return False 
    
def load_neuron_files(neuron_num,file_num_list,show_print=False,data_dir=None):
    """
    This function loads and returns data specified by the user.  
    
    :param neuron_num: nnumber of neuron we want to load, integer from 1 to 12.
    :param file_num_list: list of indices ranging from and including 1 to 99 corresponding to specific stim/sweep files we want to load.
    :param show_print: bool determining whether we print the stim files loaded (the corresponding voltage files are also loaded, just not printed).
    :keyword data_dir: path of DataTextFiles directory 
       
    Usage: 
        
    >>> [file_id_list,input_list,voltage_list,time_delta] = load_neuron_files(1,[1,2,3])
    Loaded new data: stim1.txt
    Loaded new data: stim2.txt
    Loaded new data: stim3.txt
    """
    
    print "Loading data..."
    
    if data_dir == None:
        data_dir = DATA_DIR
        
    subdir = os.path.join(data_dir,str(neuron_num))
    
    time_delta = 0.0001
    membrane_voltage_list = []
    input_current_list = []    
    file_id_list = []
    
    for ind in file_num_list:
    
        volt_txt_file = os.path.join(subdir,"sweep" + str(ind) + ".txt")
        stim_txt_file = os.path.join(subdir,"stim" + str(ind) + ".txt")
        
        if not os.path.exists(volt_txt_file) or not os.path.exists(stim_txt_file):
            warnings.warn("File not not found!")
            continue
        
        membrane_voltage = loadtxt(volt_txt_file)
        input_current = loadtxt(stim_txt_file)
        
        membrane_voltage_list.append(membrane_voltage)
        input_current_list.append(input_current)  
        new_txt_file = os.path.split(stim_txt_file)[-1]
        file_id_list.append(new_txt_file)
            
        if show_print:
            print "Loaded new data: " + new_txt_file 
            
    return file_id_list,input_current_list,membrane_voltage_list,time_delta
    
    
def load_neuron_data(neuron_num,input_type="all",max_file_ct=np.Inf,data_dir=None,verbose=False): 
    """
    This function loads and returns a subset of the data as defined by the 
    input parameters. 
    If the user want to load *specific* files, the user should use 
    :func:`load_neuron_files`.
    
    :param neuron_num: number of neuron we want to load, integer from 1 to 12
    :param input_type: input type is allowed to be "noise_only", "no_noise_only", or "all"
    :param max_file_ct: maximum number of I/O files we are allowed to load
    :keyword data_dir: path of DataTextFiles directory 
    :returns: list of input arrays, list of voltage arrays, list of file paths used 
    
    If input_type is anything else, then load all data.  
        
    Usage: 
    
    >>> [file_id_list,input_list,voltage_list,time_delta] = load_neuron_data(neuron_num=1,input_type="noise_only",max_file_ct=5)
    Loaded new data: stim13.txt
    Loaded new data: stim14.txt
    Loaded new data: stim15.txt
    Loaded new data: stim16.txt
    Loaded new data: stim17.txt
    
    This last example shows that the first 12 stimulus files were filtered because they 
    were not noisy inputs and the user specified to only load noisy inputs. 
    
    Here the user still requests 5 sweeps of data but does not require the inputs to be noisy:
    
    >>> [file_id_list,input_list,voltage_list,time_delta] = load_neuron_data(neuron_num=1,input_type="all",max_file_ct=5)
    Loaded new data: stim1.txt
    Loaded new data: stim2.txt
    Loaded new data: stim3.txt
    Loaded new data: stim4.txt
    Loaded new data: stim5.txt
    """   
    
    print "Loading data..."
    
    if data_dir == None:
        data_dir = DATA_DIR
        
    subdir = os.path.join(data_dir,str(neuron_num))
    
    # the data has comes with the package has this time step, in seconds
    time_delta = 0.0001

    if not input_type: 
        input_type = "all"
        
    membrane_voltage_list = []
    input_current_list = []    
    file_id_list = []
            
    # number of input/output pairs in the directory 
    io_ct = int(len(os.listdir(subdir))/2.0)
    ind = 0
    
    while ind + 1 < io_ct and len(file_id_list) <  max_file_ct:
    
        ind += 1
        volt_txt_file = os.path.join(subdir,"sweep" + str(ind) + ".txt")
        stim_txt_file = os.path.join(subdir,"stim" + str(ind) + ".txt")
        
        if not os.path.exists(volt_txt_file) or not os.path.exists(stim_txt_file):
            warnings.warn("File not not found!")
            continue
        
        membrane_voltage = loadtxt(volt_txt_file)
        input_current = loadtxt(stim_txt_file)
        noise_bool = is_noise(input_current) 
        
        # if the data is not the type of data we want to restrict ourselves to
        if input_type == "noise_only" and not noise_bool: 
            continue
        elif input_type == "no_noise_only" and noise_bool: 
            continue
        
        membrane_voltage_list.append(membrane_voltage)
        input_current_list.append(input_current)  
        new_txt_file = os.path.split(stim_txt_file)[-1]
        file_id_list.append(new_txt_file)
        
        if verbose:
            print "Loaded new data: " + new_txt_file
            
    return file_id_list,input_current_list,membrane_voltage_list,time_delta

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    