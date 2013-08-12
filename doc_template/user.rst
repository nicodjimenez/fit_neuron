Installation Guide
=====================
This guide will walk you through the steps necessary to get 
the package up and running

Quick Start
----------------

 #. Install the source code with pip::
    
		sudo pip install fit_neuron
       
 #. Download the data::
 
		sudo python -m fit_neuron.data.dl_neuron_data
		
 #. Check that that the output produced was similar to the following:: 
 
		Zip file not found at: /usr/local/lib/python2.7/dist-packages/fit_neuron-0.0.3-py2.7.egg/fit_neuron/data/DataTextFiles.zip
		Downloading: https://xp-dev.com/svn/neuro_fit/fit_neuron/data/DataTextFiles.zip
		Please be patient: this may take up to 20 minutes!
		Zip file successfully downloaded to: /usr/local/lib/python2.7/dist-packages/fit_neuron-0.0.3-py2.7.egg/fit_neuron/data/DataTextFiles.zip
		Unzipping data files for the first time...
		Done unzipping data files!
		Unzipped directory: /usr/local/lib/python2.7/dist-packages/fit_neuron-0.0.3-py2.7.egg/fit_neuron/data/DataTextFiles

 #. Run the tests:
     
	>>> import fit_neuron
	>>> fit_neuron.tests.run_single_test()
	 		
 		 
Required Dependencies
-------------------------

 * `NumPy <http://wiki.scipy.org/Tentative_NumPy_Tutorial>`_
 * `SciPy <http://www.scipy.org/>`_
 * `MatPlotLib <http://matplotlib.org/>`_

	
