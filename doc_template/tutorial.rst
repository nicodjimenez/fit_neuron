Tutorial
=====================

.. Contents:: 

This guide will show you how :mod:`fit_neuron` can be used to estimate 
a model from raw data, and how this model can then be evaluated against the 
raw data it is supposed to fit.

Basic Approach 
----------------------

We briefly summarize the basic approach taken by this package to estimate 
models from data and then evaluate them.  

Estimating Neurons
^^^^^^^^^^^^^^^^^^^^^^^^^^

The approach taken by the `fit_neuron` package to fitting neurons is shown 
in the following diagram:

.. tikz:: [node distance = 4cm, auto]
	\node [rectangle, draw, fill=blue!20, 
    text width=7em, text centered, rounded corners, minimum height=4em] (data) {Input/Output Data};
	\node [rectangle, draw, fill=blue!20, 
    text width=7em, text centered, rounded corners, minimum height=4em,right of=data] (fcn) {Optimization Function};
	\node [rectangle, draw, fill=blue!20, 
    text width=7em, text centered, rounded corners, minimum height=4em,below of=fcn] (options) {User Options};
	\node [rectangle, draw, fill=blue!20, 
    text width=7em, text centered, rounded corners, minimum height=4em,right of=fcn] (object) {Model Object};
    \draw [->,thick] (data) -- (fcn);
    \draw [->,thick] (options) -- (fcn);
    \draw [->,thick] (fcn) -- (object);

The computations done by `Optimization Function` shown above are implemented by the :func:`fit_neuron.optimize.fit_gLIF.fit_neuron` function, 
which wraps methods for spike processing, subthreshold estimation, and threshold estimation into a single function 
that returns a :class:`fit_neuron.optimize.neuron_base_obj.Neuron` object.  

Using Model Objects 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A model object is used much in the same way an experimentalist interacts with a patch-clamped neuron.  
The key method used is the :meth:`fit_neuron.optimize.neuron_base_obj.Neuron.update` method, which 
uses the current value of the input current injection to update the state of the neuron by a time step :math:`dt`.  
The value returned is the new value of the membrane voltage.     

.. tikz:: [node distance = 4cm, auto]
	\node [rectangle, draw, fill=blue!20, 
    text width=7em, text centered, rounded corners, minimum height=4em] (input) {$I_e(t)$};
	\node [rectangle, draw, fill=blue!20, 
    text width=7em, text centered, rounded corners, minimum height=4em,right of=input] (fcn) {model.update()};
	\node [rectangle, draw, fill=blue!20, 
    text width=7em, text centered, rounded corners, minimum height=4em,right of=fcn] (return) {$V(t+\Delta t)$};
    \draw [->,thick] (input) -- (fcn);
    \draw [->,thick] (fcn) -- (return);

.. note:: 
	There is no explicit method to determine whether the neuron is currently spiking. 
	The convention used is that the model returns a :mod:`numpy` typed value of :math:`V = \text{NaN}`
	whenever the neuron is spiking.  
	
Creating Neurons from Scratch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Typically, a user will estimate models from data using the :func:`fit_neuron.optimize.fit_gLIF.fit_neuron` function 
(which can be saved for future use using :mod:`pickle`) and hence does not need to know how to instanciate model instances 
from parameters only.  The following plot and source code illustrates how a neuron can be loaded directly from 
parameter arrays.  The meaning of the parameters is explained in :ref:`subthresh_overview` and :ref:`thresh_overview`.

.. plot:: 

	from fit_neuron.optimize import Voltage, StochasticThresh, sic_lib, Neuron
	from numpy import array,zeros, arange
	import pylab 
	
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

Running a Test Script 
-----------------------------

To run a test script which estimates a model from data, execute the following
at the command line:: 
	
	python -m fit_neuron.tests.test

The contents of this script can be viewed at :mod:`fit_neuron.tests.test` and document
much of this package's functionality.  The script will estimate the parameters for 
a neuron, save the parameters in a JSON file, save a model instance with :mod:`pickle`, plots and saves
simulation figures, and plots and saves evaluation figures.   
    
.. note:: 
	By default, :func:`fit_neuron.tests.test.run_single_test` will save the output figures 
	and data to a new directory *test_output_figures* located in the current directory.
    
Some Simulation Figures
---------------------------

Fitting results for neuron_1: 

.. image:: neuron_1/figures/stim14_rep0.png
   :height: 400px
   :width: 600px	
	
	
Another Monte Carlo simulation: 

.. image:: neuron_1/figures/stim14_rep1.png
   :height: 400px
   :width: 600px	

.. note:: 
	The green dotted lines represent the times when the model neuron spiked.

Some Statistics Figures
---------------------------

Here are some figures showing values of the Gamma coincidence factor 
for different values of :math:`\Delta t`.

Fitting results for neuron_1: 

.. image:: neuron_1/stats/gamma_factor_stim14_rep0.png
   :height: 200px
   :width: 300px	


Another Monte Carlo simulation: 

.. image:: neuron_1/stats/gamma_factor_stim14_rep1.png
   :height: 200px
   :width: 300px	

Here are some figures showing values of the Schrieber similarity measure 
for different values of the bandwidth of the Gaussian kernel :math:`\sigma`.

Fitting results for neuron_1: 

.. image:: neuron_1/stats/schrieber_similarity_stim14_rep0.png
   :height: 200px
   :width: 300px	

Another Monte Carlo simulation: 

.. image:: neuron_1/stats/schrieber_similarity_stim14_rep1.png
   :height: 200px
   :width: 300px	
