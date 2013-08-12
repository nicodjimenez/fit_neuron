Developer Guide
===============
This guide is a resource for contributing to the DiPDE Neural Simulator.  It is maintained by the Allen Institute for Brain Science.

Quick Start
-----------

 #. Get the source code.
 #. Unpack the source code.
    ::
        tar -x -v --ungzip -f dipde-0.2.2.tar.gz
 #. Install using setuptools
    ::
        cd dipde
        python setup.py install
 #. Run the tests with coverage
    :: 
        nosetests --quiet --with-coverage --cover-package dipde
        coverage -r | tee coverage.out.log | grep '^\w' | tee coverage.short.out.log
        
 #. Rebuild from source
    ::
        cd ..
        ./rebuild.sh
 		
 	
Editing Documentation
---------------------
Static documentation is in .rst files under the doc_template directory.  Add your new file (without the .rst extension) to the toclist in index.rst.
The document syntax is `reStructuredText <http://sphinx-doc.org/rest.html#rst-primer>`_ and is rebuilt by the rebuild.sh script.
The generated html documentation can be found at doc/index.html


For API docstring documentation, refer to `PEP-0257 <http://www.python.org/dev/peps/pep-0257>`_ for conventions.
Please document all important packages, classes and methods.


Quick Links
-----------
 * `Python Testing: Nose Introduction <http://pythontesting.net/framework/nose/nose-introduction>`_
 * `Sphinx Tutorial <http://sphinx-doc.org/tutorial.html>`_
 * `An Example PyPi Project <http://pythonhosted.org/an_example_pypi_project/_downloads/an_example_pypi_project.pdf>`_
 * `Easy and Beautiful Documentation with Sphinx <https://www.ibm.com/developerworks/library/os-spinx-documentation>`_
 * `T+1: Some Notes on Nosetests and Coverage <http://blog.tplus1.com/blog/2009/05/13/some-notes-on-nosetests-and-coverage>`_
 * `Documenting matplotlib <http://matplotlib.org/devel/documenting_mpl.html>`_

 
Required Dependencies
---------------------

 * `NumPy <http://wiki.scipy.org/Tentative_NumPy_Tutorial>`_
 * `SciPy <http://www.scipy.org/>`_
 * `MatPlotLib <http://matplotlib.org/>`_

	
Optional Dependencies
---------------------

 * `nose: is nicer testing for python <https://nose.readthedocs.org/en/latest>`_
 * `coverage <http://nedbatchelder.com/code/coverage>`_
	
