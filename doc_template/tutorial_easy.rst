Tutorial (easy)
=====================

The following script will show you the easy way to use :mod:`fit_neuron` to fit a model 
to data and then make voltage trace predictions.  The script loads data, fits the model,
makes predictions, then plots the predictions against the recorded data.  

For more information on the model object used, see: :class:`fit_neuron.models.gLIF.gLIF_model`.
For more information on loading data, see :func:`fit_neuron.data.data_loader.load_neuron_data`. 

.. literalinclude:: test_model.py
    :language: python

.. plot:: 

	import numpy as np
	import pylab 
	from fit_neuron.data import load_neuron_data
	from fit_neuron.models import gLIF_model

	def easy_test():
		
		# instanciate model
		model = gLIF_model()
		
		# load data 
		(file_id_list,X_list,Y_list,dt) = load_neuron_data(1,input_type="noise_only",max_file_ct=4) 
		
		# fit model
		model.fit(X_list, Y_list, dt)
		
		# predict data via model
		Y_pred_list = model.predict(X_list)
		
		# plot predicted data vs actual data for the first voltage trace in X_list
		t_arr = dt * np.arange(len(Y_pred_list[0]))
				
		pylab.plot(t_arr,Y_list[0],color='blue',label='True voltage')
		pylab.plot(t_arr,Y_pred_list[0],color='green',label='Predicted voltage')
		pylab.legend()
		
		pylab.xlabel("Time (s)")
		pylab.ylabel("Voltage (mV)")
		pylab.title("Response to Input Current")
		pylab.show() 
		
	if __name__ == '__main__':
		easy_test()


