Developer Guide
===============
This guide outlines the steps one should take to: 

#. Contribute to the **fit_neuron** package on GitHub (see https://github.com/nicodjimenez/fit_neuron).
#. Develop and tweak the source code locally for personal use. 
#. Get the latest development version.  

Quick Start
-----------

#. 	Download development version::

		git clone https://github.com/nicodjimenez/fit_neuron.git

#. 	Download test data::

		cd fit_neuron
		python -m fit_neuron.data.dl_neuron_data
	
	This will download the test data to the local directory created by step 1.  
	This directory will be different from the directory used when a user uses ``sudo pip install fit_neuron`` 
	to install the source code.  
	In either case, the ``fit_neuron.data.dl_neuron_data`` script installs the 
	test data in the same directory as the :mod:`fit_neuron.data` module found in the 
	current python path.  If the user uses ``pip`` to install ``fit_neuron``, the path to the
	:mod:`fit_neuron.data` module will be located in a ``dist-packages`` system directory 
	reserved to warehouse all third party python modules.  On the other hand, if the user uses 
	``git clone`` to install ``fit_neuron``, the :mod:`fit_neuron.data` module will simply be located in the 
	``fit_neuron/data`` directory.  

#. 	Run test script:: 

		python -m fit_neuron.tests.test
	
#. 	Build documentation::

		./create_docs.sh

.. note::
	For this script to work properly, the test script must have been run, and the 
	``sphinx-apidoc`` program must be installed. 
 		
 	
Editing Documentation
---------------------
Static documentation is in .rst files under the doc_template directory.  Add your new file (without the .rst extension) to the toclist in index.rst.
The document syntax is `reStructuredText <http://sphinx-doc.org/rest.html#rst-primer>`_.  The documentation is built by the ``create_docs.sh`` script.
The generated html documentation can be found at docs/_build/html/index.html

Quick Links
-----------
 * `Sphinx Tutorial <http://sphinx-doc.org/tutorial.html>`_
 * `An Example PyPi Project <http://pythonhosted.org/an_example_pypi_project/_downloads/an_example_pypi_project.pdf>`_
 * `Easy and Beautiful Documentation with Sphinx <https://www.ibm.com/developerworks/library/os-spinx-documentation>`_
