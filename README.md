# CNODE  
This repo contains my implementation of:  
- the Neural ODE training algorithm outlined in the 2025 conference paper ["Training Neural ODEs Using Fully Discretized Simultaneous Optimization"](https://arxiv.org/abs/2502.15642) (Mariia Shapovalova, Calvin Tsay). It reframes Neural ODE training as a large-scale NLP discretized via collocation, which is solved using IPOPT. See src/models/simultaneous.
- an updated version of the algorithm which adds an integrated residual term to the NLP objective. This is outlined in the 2025 paper ["Reliable Solution to Dynamic Optimization Problems using Integrated Residual Regularized Direct Collocation"](https://arxiv.org/pdf/2503.09123) (Yuanbo Nie, Eric C. Kerrigan).  
- a simple batched Neural ODE training algorithm with the Alternating Direction Method of Multipliers (ADMM) algorithm introduced in the 1976 paper ["A DUAL ALGORITHM FOR THE SOLUTION OF NONLINEAR VARIATIONAL PROBLEMS VIA FINITE ELEMENT APPROXIMATION"](https://www.sciencedirect.com/science/article/pii/0898122176900031?via%3Dihub) (Daniel Gabay, Bertrand Mercier) and outlined in a [textbook](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf) (Stephen Boyd et al.). 
- baseline sequential (solver-in-the-loop) Neural ODE training algorithms in PyTorch. Uses stochastic mini-batching with Adam optimization. See src/models/sequential. 

Main benchmark is a synthetic cell culture batch reaction dataset (data/bioreactor_sim_by_noise). Each dataset contains 50 time series at a given noise level. The state variable tracks 7 different concentrations:
- Glucose (G): primary carbon/energy source consumed by cells. 
- Dissolved Oxygen (O): oxygen available for cell metabolism. 
- Viable Cells (X): live, actively growing cell concentration. 
- Dead Cells (Xd): non-viable cell concentration. 
- Product (P): target bioproduct being produced. 
- Lactate (L): metabolic byproduct of glucose consumption. 
- CO2: dissolved carbon dioxide, a metabolic waste product. 

