====================================
A Guide to the Optimize Package
====================================

.. Contents::

.. _overview:

Overview
-------------------

The optimize package supports parameter estimation of stochastic generalized integrate
and fire (gLIF) neurons from patch clamp data.  In this document, we shall describe the 
following: 

	* Model equations.
	* Object oriented implementation.
	* Optimization routines.  

Model Summary
------------------

The neurons estimated by the optimize packages consist of two components:

* Subthreshold dynamics, which capture the dynamics of the membrane potential, its
  dependence on the input current, and its dependence on spike times.  The subthreshold
  dynamics are implemented by :class:`fit_neuron.optimize.subthreshold.Voltage`. 
* Threshold dynamics, which predicts the probability of a spike occuring 
  depending on the current voltage and the past history of the neuron.
  The threshold dynamics are implemented by :class:`fit_neuron.optimize.threshold.StochasticThresh`.

Subthreshold Equations
--------------------------

The following difference equation defines the subthreshold model: 

.. math::
	V(t + dt) - V(t) = \alpha_1 + \alpha_p V + \alpha_{I_e} I_e + \alpha_g g(V) + \sum_{i=1}^{i=n} \beta_i I_i(t,\{\hat{t}\},V)
	
where 

* :math:`t` is the current time
* :math:`dt` is the time step, typically corresponding to the time step in the recorded patch clamp data
* :math:`\{\hat{t}\}` is the set of previous spike times
* The :math:`I_i` are the *spike triggered currents* or *spike induced currents* which depend only on the spiking 
  history of the neuron, and optionally on the current value of the voltage :math:`V`.
* The function :math:`g(V)` is a voltage nonlinearity function that allows the model to have
  an upward spike initiation.  Commonly used voltage nonlinearities are the quadratic and exponential functions.

See :class:`fit_neuron.optimize.threshold.StochasticThresh` for the implemenation of this difference equation.

Whenever a spike occurs (as determined by the threshold equations), the state of the neuron is updated
as follows: 

.. math:: 
	V \gets V_r
	
	I_i \gets \phi(I_i), \quad i = 0,\hdots, l 
	
	t \gets t + t_{ref}

where 

* :math:`V_r` is the reset voltage.
* :math:`t_{ref}` is the refractory period of the neuron. 
* :math:`\phi` is an optional *update rule*.    

This update rule is implemented for general cases of spike induced currents 
by :meth:`fit_neuron.optimize.subthreshold.Voltage.spike`.

The Optimize package allows two variations of spike triggered currents.

Gerstner Adaptation Currents 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In [MS2011]_ the spike triggered currents take the form of a sum of step functions.
This sum can be written as:

.. math:: 
	I_i(t,\{\hat{t}\}) = \sum_{\hat{t} \in \{\hat{t}\}} I_{[0,a_{i} ]} (t - \hat{t})

where the indicator functions :math:`I_{[0,a_{i} ]}` are defined as follows:

.. math:: 
    I_{[0,a_{i}]}(x) =
    \begin{cases}
      0, & \text{otherwise} \\
      1, & \text{if}\ x \in [0,a_{i}]
    \end{cases}

By looking at the difference equation for the voltage, one sees that 
the parameter vector :math:`{ \bf \beta } = [\beta_0,\hdots,\beta_n]^{\top}` defines 
the *shape* of the spike triggered currents, and the :math:`\{a_i\}` parameters 
define the intervals during which the shape is constant.  

In this case, the update rule simply appends the current spike to the 
spiking history :math:`\{\hat{t}\}`:

.. math::
	\{\hat{t}\} \gets \{\hat{t}\} \cup t 

The Gerstner adaptation currents are implemented in :class:`fit_neuron.optimize.sic_lib.StepSic`.

.. note:: 
	The equations are written here in a form that matches the implemenation.  The equations are 
	written differently in [MS2011]_ but are perfectly equivalent up to a linear transformation 
	of the parameter vector :math:`\beta`. 


Mihalas-Niebur Adaptation Currents
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
An alternative form of spike triggered currents is used in [MN2009]_ and consists of exponentially 
decaying currents with an additive reset.  The equations are as follows: 

.. math:: 
	\frac{dI_i}{dt} = -k_i I_i 

and the reset equation, applied whenever the neuron spikes, is:

.. math:: 
	\phi(I_i) = I_i + 1 

The Mihalas-Niebur adaptation currents are implemented in :class:`fit_neuron.optimize.sic_lib.ExpDecay_sic`.

Object Oriented Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When calling the optimization routine :func:`fit_neuron.optimize.fit_gLIF.fit_neuron`, the user has the ability to specify
any spike induced object he/she would like, as long as the user defines the class 
of the spike triggered current to inherit from the following abstract class: :class:`fit_neuron.optimize.sic_lib.SicBase`.

.. _thresh_eq:

Threshold Equations 
-------------------------

The stochastic neuron has the following *hazard rate*: 

.. math::
	h(t,V) = \exp \left( c_0 + c_1 V + \sum_{\hat{t} \in \{\hat{t}\}} \sum_{i=1}^{i=m} d_i I_{[0,b_i]} (t-\hat{t}) + \sum_{j=1}^{j=l} e_j Q_j(t) \right)
	
where the :math:`I_[0,b_i]` parameters are the indicator variables (see above), and 
the :math:`Q_j` parameters are probability currents which shall be referred to as 
*voltage chasing currents*.  These currents give the stochastic spike emission process a component
that adapts to the history of the voltage.  The equations used for the voltage chasing currents 
are: 

.. math:: 
	\frac{dQ_i}{dt} = r_i (V - Q_i)
	
When the neuron spikes, the voltage chasing currents are set to the reset potential: 

.. math:: 
	Q_i \gets V_r
	
The hazard rate is computed at each time step and compared to a uniformly distributed random number to 
determine whether the neuron spikes here.  This computation is performed by 
:meth:`fit_neuron.optimize.threshold.StochasticThresh.update_X_arr`.   
	
Fitting Procedure Overview
-------------------------------

The parameter estimation algorithm provided by the :func:`fit_neuron.optimize.fit_gLIF` function proceeds as follows: 

#. Extract spikes and spike shapes from the raw data.
#. Take the voltage traces with the spike shapes removed, and estimate the subthreshold parameters by linear regression of the 
   derivative of the voltage.
#. Simulate the model neuron using the same inputs as the raw data, and force the spikes to happen at the times
   the biological neuron was observed to spike.  This will produce simulated voltage traces. 
#. Fit the threshold parameters such that these parameters maximize the log likelihood of the obseved 
   spike trains being emitted by the simulated voltage traces.
   
.. _subthresh_overview:

Subthreshold Fitting Procedure Overview
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The subthreshold parameters are obtained via linear regression to 
the observed voltage differences.  

The equation that is solved is the following:

.. math:: 
	\min_{b} \|Xb - Y\|^2
	
where 

.. math:: 
	X = \begin{bmatrix}
	V(0) & 1 & I_e(0) & g(V) & I_0(0) & \hdots & I_n(0)  \\
	V(1) & 1 & I_e(1) & g(V) & I_0(1) & \hdots & I_n(1) \\
	\vdots & \vdots & \vdots & \vdots &\vdots  & \vdots  & \vdots 
	\end{bmatrix}

and 

.. math:: 
	Y = \begin{bmatrix}
	V(1) - V(0) \\ 
	V(2) - V(1) \\
	\vdots
	\end{bmatrix}
	
The value of :math:`b` that minimizes this expression is the parameter
vector chosen for the subthreshold object :class:`fit_neuron.optimize.subthreshold.Voltage`.

.. note:: 
	The values of :math:`V` used above represent the values in the recorded voltage traces.
	The values of the spike induced currents :math:`I_i(t)` are computed
	based on the recorded voltage values and the recorded spike times.  Hence the 
	estimation process resembles maximum likelihood.  

.. note::
	If no voltage nonlinearity is provided, or if it is set to :attr:`None`, the parameter 
	vector will still correspond to the :math:`b` vector above but with the voltage nonlinearity skipped.

.. _thresh_overview:

Threshold Fitting Procedure Overview
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The threshold parameters are obtained via max likelihood of the observed spike train.  Following 
along the lines of [MS2011]_ we may re-write the threshold equation in :ref:`thresh_eq` as follows: 

.. math:: 
	h(t) = \exp \left({\bf w}_t^{\top} {\bf X}_t (t) \right)

where 

.. math:: 
	{\bf X}_t (t) = [1,V(t),I_1(t),\hdots,I_m(t),Q_1(t),\hdots,Q_l(t)]^{\top}.

as computed by :meth:`fit_neuron.optimize.threshold.StochasticThresh.update_X_arr`.

The probability of there being a spike in a time increment :math:`dt` is 

.. math:: 
	p(t) = 1 - \exp \left(-h(t) dt \right) 
	
The probability of observing spikes at indices in the spiking set :math:`S`, 
and no spikes at the indices in the complement of this set :math:`S^c`, is:

.. math:: 
	p(S) = \prod_{i \in S} p(t_i) \prod_{j \in S^c} (1 - p(t_j))
	
Taking logs of both sides, we obtain 

.. math:: 
	L({\bf w}_t) = \sum_{i \in S} \log (p(t_i)) + \sum_{j \in S^c} \log(1 - p(t_j))
	
which may be approximated up to a constant as: 

.. math:: 
	L({\bf w}_t) = \sum_{i \in S} {\bf w}_t^{\top} {\bf X}_t(t_i) - \sum_{j \in S^c} \exp \left({\bf w}_t^{\top} {\bf X}_t(t_j) \right)

The :math:`k`'th elements of the gradient of this function w.r.t. :math:`{\bf w}_t` are: 

.. math:: 
	[\nabla L({\bf w}_t)]_k = \sum_{i \in S} [{\bf X}_t(t_i)]_k - \sum_{j \in S^c} [{\bf X}_t(t_j)]_k \exp \left({\bf w}_t^{\top} {\bf X}_t(t_j) \right)

The :math:`(l,m)` elements of the Hessian matrix are: 

.. math:: 
	[H({\bf w}_t)]_{l,m} = - \sum_{j \in S^c} [{\bf X}_t(t_j)]_l [{\bf X}_t(t_j)]_m \exp \left({\bf w}_t^{\top} {\bf X}_t(t_j) \right)

It is trivially seen that the Hessian matrix is negative definite.  Hence the negative log likelihood is a convex 
function of the parameters and convex optimization techniques are applicable here.  We use 
a Newton algorithm to update the values of the parameters: 

.. math:: 
	{\bf w}_t^{\text{new}} = {\bf w}_t^{\text{old}} - H^{-1}({\bf w}_t) \nabla L({\bf w}_t)

The most computationally expensive step in this process is the computation of the gradients and hessians, which 
must be done at every step.  Significant speedups can be achieved by distributing the computations of the 
gradients and the hessians to multiple processors.  This is done in :func:`fit_neuron.optimize.threshold.par_calc_log_like_update`.

References
------------------

.. [RB2005] Brette, Romain, and Wulfram Gerstner. "Adaptive exponential integrate-and-fire model as an effective description of neuronal activity." 
			Journal of neurophysiology 94.5 (2005): 3637-3642.
			
.. [MN2009] Mihalas, Stefan, and Ernst Niebur. "A generalized linear integrate-and-fire neural model produces diverse spiking behaviors." 
			Neural computation 21.3 (2009): 704-718.
			
.. [MS2011] Mensi, Skander, et al. "Parameter extraction and classification of three cortical neuron types reveals two distinct adaptation mechanisms." 
			Journal of neurophysiology 107.6 (2012): 1756-1775.
