# SHORT
Solving for Harbingers Of Rare Transitions

This repository accompanies "Learning Rare Stratospheric Transitions From Short Simulations", submitted to Monthly Weather Review
Authors: Justin Finkel, Robert J. Webber, Edwin P. Gerber, Dorian S. Abbot, Jonathan Weare

This is a work in progress. Please contact Justin Finkel at jfinkel@uchicago.edu for clarification and guidance. Below is an outline of file functionality.

The overall pipeline follows the structure of the two "driver" files, hm_bothcast_driver.py, which analyzes the Holton-Mass model as described in the paper (and more), and doublewell_driver.py, which analyzes a much simpler double-well potential for a friendlier introduction to the method. The pipeline below computes both forward and backward committors, forward and backward lead times, and potentially higher moments of integrated risk functions. Reactive currents are also computed. The steps are:
(1) Set parameters and define a model
(2) Compute a least-action pathway between the two metastable states, A and B
(3) Run a long control simulation 
(4) Run many short simulations, using the long run to seed initial conditions
(5) Perform the DGA calculations to compute the forward committor, lead time, and steady state distribution. The code also computes the backward committor and higher moments of more general risk functions relating to transition paths. 
(6) Perform LASSO regression to find the best low-dimensional proxies for the forward committor.
(7) Plot the results.

Steps (3) and (4) are very expensive for the Holton-Mass model, and we recommend starting with the cheaper doublewell potential. Below is a list of files with their basic functionality.

model_obj.py: abstract class definition, Model, for defining a bistable dynamical system with stochastic forcing. 
hm_model.py: class definition of HoltonMassModel, inherited class from Model for defining the Holton-Mass model.
doublewell_model.py: class definition of DoubleWellModel, inherited class from Model for defining a simple, two-dimensional double well potential.
hm_params.py: file with specification of input parameters for the Holton-Mass model. Parameters are grouped into two sets: physical parameters for the model, and algorithmic parameters for the Dynamical Galerkin Approximation (DGA) procedure. 
doublewell_params.py: same as hm_params.py, but for the doublewell model.
helper.py: auxiliary file with oft-used functions for averaging, plotting, and the like.
data_obj.py: class definition of Data, a class for organizing short trajectories.
function_obj.py: abstract class Function for expressing and computing forecast functions such as the committor. The only fully implemented inherited class is Markov State Models, as described in the paper, but other basis functions are possible. 
tpt_obj.py: class definition for TPT analysis, which calls the Function object to compute forward committors and other quantities of interest. This file is (mostly) agnostic to the details of the input model, so theoretically other dynamical systems could be plugged in relatively seamlessly. One would just have to implement inherited classes from Model. 

This repository is not yet optimized, but will eventually become a downloadable package. 


