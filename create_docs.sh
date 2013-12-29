#!/bin/bash
rst2html.py README.rst > README.html
rm -rf docs
mkdir docs 
cp -avr test_output_figures/neuron_1 docs/
sphinx-apidoc -F -o docs fit_neuron/
cp ./build/conf.py ./docs/conf.py
cp ./doc_template/*.rst ./docs
cp -r ./fit_neuron/tests/test_output_figures/neuron_1 docs/_build/html
cp ./fit_neuron/tests/test_model.py docs/
cd docs
make html
cd .. 
rm -rf docs/neuron_1
rm -rf docs/_build/html/figures
rm -rf docs/_build/html/json_files
rm -rf docs/_build/html/pickle_files
rm -rf docs/_build/html/stats



