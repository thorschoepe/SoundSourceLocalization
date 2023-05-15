# SoundSourceLocalization

This repository contains all the code of the publication "Closed-Loop Sound Source Localization in Neuromorphic Systems".
A preprint can be found at:
https://www.techrxiv.org/articles/preprint/Event-Based_Sound_Source_Localization_in_Neuromorphic_Systems/21493290

The repository contains three experiments to estimate the interaural time difference (ITD) with the Time Difference Encoder and a ring-attractor network. 

In the first experiment, **nest simulation**, computer generated Poisson spike trains with a predefined ITD are provided to the network presented in the article. The network is implemented in the Nest simulator. 

In a second experiment, **sweep**, the network is implemented on an AerNode board and a SpiN-3 board in a robotic setup equipped with a pan-tilt unit with a 3Dio Binaural Microphone. 
A speaker playing a 500 Hz pure tone is placed in front of the setup. The response of the system during a 180 degrees sweep of the binaural microphone is recorded.

In the third experiment,**closed loop**, the closed loop performance of the whole system is analyzed. 
The pan-tilt unit turns toward the direction of the sound source. It maintains the direction of the speaker. 

All the data and scripts of the paper are provided in this repository. 
Detailed instructions how to use the data and scripts are given in the README files in the subfolders.
