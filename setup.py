import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "fit_neuron",
    version = "0.0.4",
    author = "Nicolas D. Jimenez",
    author_email = "nicodjimenez@gmail.com",
	packages=['fit_neuron','fit_neuron.data','fit_neuron.optimize','fit_neuron.evaluate','fit_neuron.tests'],
	#packages=['fit_neuron','fit_neuron.optimize','fit_neuron.evaluate','fit_neuron.tests'],
	package_dir={'fit_neuron': 'fit_neuron'},
	#package_data={'fit_neuron': ['data/DataTextFiles.zip']},
    description = "Package for estimation and evaluation of neural models from patch clamp neural recordings.",
    license = "Apache",
    keywords = "neuron linear integrate and fire patch clamp fitting parameter estimation spike distance metrics",
    url = "http://pythonhosted.org/fit_neuron",
    long_description=read('README.rst'),
    classifiers=[
        "Development Status :: 1 - Planning",
        "License :: Freely Distributable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    scripts=['fit_neuron/data/dl_neuron_data.py']
	)

