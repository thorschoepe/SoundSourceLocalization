code to run the experiment with SpiNNaker and FPGA. 

Load the bitfile "ed_of_robot_control_top.bit" onto the AERnode board connecting SpiNNaker and the motors.
Load the bitfile nas_tde_network_spinn_top.bit onto the AERnode board containing the NAS and the TDEs.
Run nastde_closed_loop_2022_02_22.py on a SpiN-3 board with the SpyNNaker version 4.0.0. and python 2.7.

Raw data of the runs are stored at ../250Hz, ../500Hz and ../speech.

The angle of the pan-tilt unit can be extracted from the movies at ../250Hz, ../500Hz and ../speech 
with the script "estimate_angle.py". The script "estimate_velocity.py" calculates the velocities. 
The script "network_rasterplot.py" plots Figure 11 in the article 
"Closed-Loop Sound Source Localization in Neuromrophic Systems". 

