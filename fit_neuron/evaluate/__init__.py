"""
evaluate
~~~~~~~~~~~~

This package deals with the evaluation of the parametrized models
returned by optimization routines.  Evaluation criteria used are: 

1.  Voltage mean square error.  
2.  Spike distance metrics.
    
    a.  Gamma coincidence factor - :func:`spkd_lib.gamma_factor`
    b.  Victor Papura spike distance - :func:`spkd_lib.victor_purpura_dist`
    c.  van Rossum distance - :func:`spkd_lib.van_rossum_dist`
    d.  Schreiber et al. similarity measure - :func:`spkd_lib.schrieber_sim`
    
"""

from spkd_lib import *
from evaluate import *



