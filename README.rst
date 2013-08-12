=============
fit_neuron
=============

**fit_neuron** is a python package for the fast estimation of generalized integrate and fire neural models 
from patch clamp electrophysiological recordings.  The optimization routines implements a fitting procedure 
described in [RB2005]_ and [MS2011]_.  These routines can estimate the models described in [RB2005]_, [MN2009]_, and [MS2011]_.  
As described in depth in the documentation, the subthreshold 
parameters are estimated using linear regression and the threshold parameters are estimated 
using maximum likelihood.  The fitting routine is built for speed: it estimates neuron parameters for 10 seconds of data 
in about 50 seconds on a quad core Asus laptop.  *fit_neuron* also contains efficient implementations 
of the following spike distance measures: Victor-Purpura [DA2003]_, van Rossum [VR2001]_, Schrieber [SS2003]_, and Gamma [RJ2008]_
which can be used to evaluate the accuracy of estimated models, as well as provide measures 
of synchrony between spike trains.    

:Date: 2013-08-10
:Version: 0.0.3
:Authors: - Nicolas D. Jimenez

Links 
----------

1) **Pypi** 

The latest stable version is available to download at: https://pypi.python.org/pypi/fit_neuron.

2)  **GitHub**

The latest development version is available at: https://github.com/nicodjimenez/fit_neuron.  All relevant contributions are welcome 
and fast review of pull requests is guaranteed.  

3)  **Documentation**   

Sphinx documentation is available at: http://pythonhosted.org/fit_neuron/.


Dependencies
-------------

1) **Numpy** 

The standard python module for matrix and vector computations: https://pypi.python.org/pypi/numpy.

2) **Scipy** 

The standard python module for statistical analysis: http://www.scipy.org/install.html.

3) **Matplotlib**

The standard python module for data visualization: http://matplotlib.org/users/installing.html.

Installation 
-----------------------

The fit_neuron package can be installed as follows::

	sudo pip install fit_neuron
	

The data for the fit_neuron package is then installed as follows::

	sudo python -m fit_neuron.data.dl_neuron_data
	
	
.. warning:: 
	Running this script for the first time will download a 300 MB zip file containing test recordings 
	which is then unzipped to over 1 GB of text files in the installation directory of the *fit_neuron* 
	package.  This may take up to 20 minutes depending on your bandwidth.  After the files are downloaded, the test 
	data will be easily accessible via the *fit_neuron.data* package.  

	
Testing
------------

The main testing script for fit_neuron can be run as follows:: 

	python -m fit_neuron.tests.test


This will create a directory called *test_output_figures* in the current directory.  

Feel free to contact me at nicodjimenez [at] gmail.com if you have any questions / comments.  

References
------------------

.. [RB2005] Brette, Romain, and Wulfram Gerstner. "Adaptive exponential integrate-and-fire model as an effective description of neuronal activity." 
			Journal of neurophysiology 94.5 (2005): 3637-3642.
			
.. [MN2009] Mihalas, Stefan, and Ernst Niebur. "A generalized linear integrate-and-fire neural model produces diverse spiking behaviors." 
			Neural computation 21.3 (2009): 704-718.
			
.. [MS2011] Mensi, Skander, et al. "Parameter extraction and classification of three cortical neuron types reveals two distinct adaptation mechanisms." 
			Journal of neurophysiology 107.6 (2012): 1756-1775.

.. [RJ2008] Jolivet, Renaud, et al. "A benchmark test for a quantitative assessment of simple neuron models." 
			Journal of neuroscience methods 169.2 (2008): 417-424.
			
.. [SS2003] Schreiber, S., et al. "A new correlation-based measure of spike timing reliability." 
			Neurocomputing 52 (2003): 925-931.
			
.. [VR2001] van Rossum, Mark CW. "A novel spike distance." 
			Neural Computation 13.4 (2001): 751-763.
			
.. [DA2003] Aronov, Dmitriy. "Fast algorithm for the metric-space analysis 
			of simultaneous responses of multiple single neurons." Journal 
			of Neuroscience Methods 124.2 (2003): 175-179.

