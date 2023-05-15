import numpy as np
import time
import glob
import os
import argparse
from scipy.spatial import distance
import matplotlib.pyplot as plt
import math
from quantities import ms
import matplotlib.patches as mpatches

def rasterplot(spiketimes,populations_size, simtime, x_labels, y_labels, titles, path_save, colors, types, binnumber):
	
	color = [ 'midnightblue', 'maroon', 'mediumblue', 'brown', 'blue', 'red', 'dodgerblue',  'salmon', 'orange']
	number_subplots = len(spiketimes)
	ax = [[] for _ in range(number_subplots+1)]
	
	
	fig = plt.figure(figsize =(8,3.5), dpi=300)
	plt.rc("font", size=7)
	

	
		
	
	for num in range(1, number_subplots+1):
		ax[num] = fig.add_subplot(int((number_subplots+1)/4),4, num)
		if num > 3:
			ax[num].set_xlabel(x_labels)
		ax[num].set_ylabel(y_labels[num-1])
		ax[num].set_title(titles[num-1])
		
	
		
		
		if types[num-1] == 'raster':
			ax[num].set_xlim(simtime)
			if populations_size[num-1] < 10:
				ax[num].set_yticks(range(0,10,1))
			else:
				ax[num].set_yticks(np.arange(0,populations_size[num-1],populations_size[num-1]/5))
			ax[num].set_ylim([-0.25,populations_size[num-1]-0.75])
			if spiketimes[num-1] != Spikes.tde_in_right and spiketimes[num-1] != Spikes.tde_in_left:
				for i in range(len(spiketimes[num-1].segments[0].spiketrains)):
					ax[num].scatter(np.array(spiketimes[num-1].segments[0].spiketrains[i])/1000.0-10, [i]* len(spiketimes[num-1].segments[0].spiketrains[i]), s=100, color = colors[num-1], marker='|')
			else:
				for i in range(len(spiketimes[num-1])):
					ax[num].scatter(np.array(spiketimes[num-1][i])/1000.0-10, [i]* len(spiketimes[num-1][i]), s=100, color = colors[num-1], marker = '|')
		elif types[num-1] == 'hist':
			n, bins, patches = ax[num].hist(np.array(spiketimes[num-1].segments[0].spiketrains[0])/1000.0-10, int(binnumber),range=(simtime), density=False, alpha = 1.0, color = 'w')
			ax[num].plot(np.arange(0,simtime[1],simtime[1]/binnumber)+simtime[1]/binnumber/2,np.array(n)*binnumber/simtime[1], color=colors[num-1])
			ax[num].plot(np.arange(0,simtime[1],simtime[1]/binnumber)+simtime[1]/binnumber/2,np.array(n)*binnumber/simtime[1], color=colors[num-1], marker='.')
			ax[num].set_ylim([0,500])
			ax[num].set_xlim(simtime)
		elif types[num-1] == 'hist_multi':
			for i in range(len(spiketimes[num-1])):
				n, bins, patches = ax[num].hist(np.array(spiketimes[num-1][i])/1000.0-10, int(binnumber),range=(simtime), density=False, alpha = 1.0, color = 'w')
				ax[num].plot(np.arange(0,simtime[1],simtime[1]/binnumber)+simtime[1]/binnumber/2,np.array(n)*binnumber/simtime[1], color=colors[num-1][i])
				ax[num].plot(np.arange(0,simtime[1],simtime[1]/binnumber)+simtime[1]/binnumber/2,np.array(n)*binnumber/simtime[1], color=colors[num-1][i], marker='.')
				ax[num].set_ylim([0,350])
				ax[num].set_xlim(simtime)
		elif types[num-1] == 'angle':
			ax[num].scatter(np.array(range(len(angle)))/25.0, 175-np.degrees(angle), s=100, color = colors[num-1], marker='.')
			ax[num].set_xlim([0,6])
			ax[num].grid(True)


			
	fig.tight_layout()
		
	fig.savefig(path_save)

class Size:
	fpga = 512
	e_pg = p_en = 64 
	edge_dvs = 128
	pop_dvs = edge_dvs**2
	fpga = 512

class Spikes:
	tde_in=0
	out_r=0
	out_l=0 
	p_en_right=0
	p_en_left=0
	e_pg_right=0
	e_pg_left=0
	onset=0
	gi=0
	tde_in_right=0
	tde_in_left = 0

num_tdes = 4 
angle = np.load('/home/spinnaker/Documents/projects/Sevilla_2021_TDE_SSL/data/final/closed_loop/500Hz/500_09.mov_angle.npy')
Spikes.p_en_right, Spikes.p_en_left, Spikes.e_pg_right, Spikes.e_pg_left, Spikes.tde_in, Spikes.out_r, Spikes.out_l, Spikes.onset, Spikes.gi = np.load('/home/spinnaker/Documents/projects/Sevilla_2021_TDE_SSL/data/final/closed_loop/500Hz/sweep_frequ500_datetime_22_02_2022_18-27-25.npy')
Spikes.tde_in_right = Spikes.tde_in.segments[0].spiketrains[0:4]
Spikes.tde_in_left = Spikes.tde_in.segments[0].spiketrains[4:8]

spikedata = [Spikes.tde_in_left, Spikes.out_l,Spikes.e_pg_right,angle, Spikes.tde_in_right, Spikes.out_r,Spikes.onset]
size_populations = [ 4, 4, Size.e_pg,1,1,1,Size.p_en]
y_labels = ['(Hz)', '(Hz)','(id)', '(deg)', '(Hz)','(Hz)', '(id)']
titles = ['tde left frequency', 'out left frequency', 'ring attractor bump', 'pan-tilt unit angle', 'tde right frequency', 'out right frequency', 'center detector spikes']
colors = [[ 'midnightblue', 'mediumblue', 'blue', 'dodgerblue'],'blue','black', 'orange',[ 'maroon', 'brown', 'red', 'salmon'], 'red', 'magenta']
types = ['hist_multi','hist', 'raster', 'angle','hist_multi','hist', 'raster']
path = '/home/spinnaker/Documents/projects/Sevilla_2021_TDE_SSL/data/rasterplot_closedloop2.png'
binnumber = 12.0
	
rasterplot(spikedata,size_populations, [0,6],'time (sec)', y_labels, titles, path, colors, types, binnumber)
