#! /usr/bin/env python

"""
A script that automatically downloads the test data to the directory of the 
:mod:`fit_neuron.data` package within dist-packages.  The script may be run as:

.. code-block:: none

    sudo python -m fit_neuron.data.dl_neuron_data


The sample recordings contain data for 12 neurons.  
The functions in this file use :mod:`urllib2` to download a zip file, which is then 
unzipped in the :mod:`fit_neuron.data` directory.  

The user will need to be patient the first time the user attempts to call 
the data loading functions in this file, as a 300 MB zip file will be downloaded 
and then unzipped to txt files.  The disk usage of the unzipped text files 
will be over 1 GB.  Once the sample recordings are downloaded, the 
downloading and unzipping of the data will be skipped.   

.. note:: 
    While running this script is the easiest way for a user to start testing 
    the fitting methods, the user may also download and unzip the data manually. 
    In this case, the path of the top level directory (DataTextFiles by default) 
    containing the data will need to be 
    specified as a keyword argument when calling :func:`fit_neuron.data.data_loader.load_neuron_files` 
    and :func:`fit_neuron.data.data_loader.load_neuron_data`.
""" 

import os
import zipfile
from urllib2 import urlopen, URLError, HTTPError
import fit_neuron.data

# current directory of this file
CUR_DIR = os.path.abspath(os.path.dirname(fit_neuron.data.__file__))

# directory where data data should be unzipped
DATA_DIR = os.path.join(CUR_DIR,"DataTextFiles")

# url containing all the data
ZIP_ADDRESS = "https://xp-dev.com/svn/neuro_fit/fit_neuron/data/DataTextFiles.zip"

# where the zip file should be located
ZIP_NAME = os.path.join(CUR_DIR,"DataTextFiles.zip")

def dl_file(url=ZIP_ADDRESS):
    """
    Downloads a zip file containing test data into the current directory.  
    """
    #print ZIP_NAME
    if not os.path.exists(ZIP_NAME):
        print "Zip file not found at: " + str(ZIP_NAME)
        try:
            f = urlopen(url)
            print "Downloading: " + url
            print "Please be patient: this may take up to 20 minutes!"
    
            # Open our local file for writing
            with open(ZIP_NAME, "wb") as local_file:
                local_file.write(f.read())
                
            print "Zip file successfully downloaded to: " + ZIP_NAME
    
        #handle errors
        except HTTPError, e:
            print "HTTP Error:", e.code, url
        except URLError, e:
            print "URL Error:", e.reason, url
        
def unzip_test_data():
    """
    Unzips the data that comes with the module.  This function 
    needs to be called in order for the :func:`load_neuron_files` and 
    :func:`load_neuron_data` functions to work.  Once the data is unzipped, 
    this function will automatically detect the existence of the folder with 
    the data and will not execute anything anymore.  
    """
    
    if not os.path.exists(DATA_DIR):
        print "Unzipping data files for the first time..."
        zip_name = os.path.join(CUR_DIR,"DataTextFiles.zip")
        zipfile.ZipFile(zip_name).extractall(CUR_DIR)
        print "Done unzipping data files!"
        print "Unzipped directory: " + str(DATA_DIR)
    else: 
        print "Directory already exists: " + str(DATA_DIR)

if __name__ == '__main__':
    dl_file()
    unzip_test_data()

    