# CNODE  
This repo contains implementations of:  
- the Neural ODE training framework outlined in the paper ["A simultaneous framework for training neural ODEs using full discretization
and large-scale nonlinear programming"](https://www.sciencedirect.com/science/article/pii/S0959152426002179?ref=pdf_download&fr=RR-2&rr=a4f18efbac8b4188). It reframes Neural ODE training as a large-scale NLP discretized via collocation, which is solved using IPOPT. See src/models/simultaneous/node_pyomo_base.
- an Alternating Direction Method of Multipliers (ADMM) extension of the training algorithm which allows training on large datasets of time series, see src/models/simultaneous/node_pyomo_admm & admm_mp.
- baseline sequential (solver-in-the-loop) Neural ODE training algorithms in PyTorch. Uses stochastic mini-batching with Adam optimization. See src/models/sequential. 

First benchmark is a simulation of a fed-batch cell culture bioreactor (data/bioreactor_simulation). 

