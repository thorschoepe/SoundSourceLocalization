Run the code tde_ssl_poisinput.py to replicate the data in "Closed-Loop Sound Source Localization in Neuromorphic Systems", Schoepe et al. 2023 
Figure 6a-c and Figure 7a-c. Make sure to include the nest linear tde model from the github branch ../../nest_tde/model. 
The data for this code are given in ../data. The zip file contains data with and without noise. 
The data with random noise has the ending n.npy (figure 6b and 7b). 
The data with random noise and normal distributed ISIs has the ending 2n.npy. 
All other data with no n at the end is noiseless. 

Run the code tde_ssl_nest_plot.py to plot the data. 
Change the ending in the data loader to plot data with and without noise.
