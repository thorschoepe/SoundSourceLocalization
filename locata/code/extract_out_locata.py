import numpy as np
import matplotlib.pyplot as plt
import math
import os, fnmatch
import scipy
import random
from quantities import ms




# function to find files at path
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


# class for loading spikes
class Spikes:
	pass	
	p_en_right=0
	p_en_left=0 
	e_pg_right=0
	e_pg_left=0
	tde_in=0
	out_r=0
	out_l=0 
	onset=0 
	gi=0

## defining variables, all variables in ms
maxtime = 30000 # duration of relevant recording time
step = 100 # step size for instantaneous rate calculation
binsize = 1000 # bin size fo instantaneous rate estimate

error = []
spikes_talking_a=[]
time_talking_a =[] 
angle_talking_a = []
time_receiv_talking_a = []

# load file names for all spikes
for recording in range(3):
	files = find('sweep_runtask'+str(3)+'_recording_'+str(recording+1)+'*', '../data/development/spikes/')
	for repetitions in range(5):
		
		
		# load spikes and ground truth data
		Spikes.p_en_right, Spikes.p_en_left, Spikes.e_pg_right, Spikes.e_pg_left, Spikes.tde_in, Spikes.out_r, Spikes.out_l, Spikes.low_r, Spikes.low_l, Spikes.onset, Spikes.gi = np.load(files[repetitions])
		spikes_out_r = [Spikes.out_r.segments[0].spiketrains[0][i]/ms for i in range(len(Spikes.out_r.segments[0].spiketrains[0]))]
		spikes_out_l = [Spikes.out_l.segments[0].spiketrains[0][i]/ms for i in range(len(Spikes.out_l.segments[0].spiketrains[0]))]
		spikes_tde_in = [Spikes.tde_in.segments[0].spiketrains[0][i]/ms for i in range(len(Spikes.tde_in.segments[0].spiketrains[0]))]
		np.save('../data/development/spikes/out_spikes_recording_'+str(recording+1)+'_'+str(repetitions+1)+'.npy',[spikes_out_r,spikes_out_l, spikes_tde_in])
		
