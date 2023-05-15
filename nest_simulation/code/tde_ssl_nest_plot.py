""" code to plot the response of the nest SSL network. Add the path to the files and the path to the location to save before running the script"""

import matplotlib as mpl
import matplotlib.pyplot as pl
import numpy as np
import logging
import random
import math
import os
import seaborn as sns
from matplotlib.colors import BoundaryNorm
import matplotlib.patches as mpatches

pathsave = ""
num_tdes = 4
tau_facs = np.arange(0.1, (float(num_tdes+1))/10.0, 0.1)
neuron_models = ["iaf_psc_lin_semd"]
tau_trig = 1.3

red_patch = mpatches.Patch(color='red', label='right NAS')
blue_patch = mpatches.Patch(color='blue', label='left NAS')
cyan_patch = mpatches.Patch(color='cyan', label='left ideal')
orange_patch = mpatches.Patch(color='orange', label='right ideal')


frequencies = np.array(range(500,1500,250))
distance = 50
num_positions = 9
num_channels = 1
minimum_channel = 25 # was 32
size_ring = 32
step = 1
cochlea_channels = 64
time_differences = 8.0 # 100 us
res_time_diff = 0.5
num_timediff = len(np.arange(-time_differences, time_differences+res_time_diff, res_time_diff))

times = np.arange(-time_differences, time_differences+res_time_diff, res_time_diff)
#print times

radial_position = ['-90deg, -880us', '-67.5deg, -810us', '-45deg, -610us', '-22.5deg, -320us','0deg, 0us', '22.5deg, 320us', '45deg, 610us', '67.5deg, 810us', '90deg, 880us']

source_distance_degrees_1 = [35, 36.6, 40.8, 46.4]
source_distance_degrees_2 = [65, 64.1, 61.52, 57.4]


	
def skipper(fname):
    with open(fname) as fin:
        no_comments = (line for line in fin if not line.lstrip().startswith('#'))
        next(no_comments, None) # skip header
        for row in no_comments:
            yield row




class settings:
	address_size = 2
	timestamp_size = 4
	on_off_both = 1
	num_channels = 64
	mono_stereo = 1

for neuron_model in neuron_models:
			for frequency in frequencies:
				spikes_position = [[] for _ in range(num_timediff)]
				spikes_position2 = [[] for _ in range(num_timediff)]
				fig1 = pl.figure(figsize =(12,8), dpi=300)
				fig2 = pl.figure(figsize =(12,8), dpi=300)
				fig3 = pl.figure(figsize =(12,8), dpi=300)
				fig4 = pl.figure(figsize =(12,8), dpi=300)
				fig5 = pl.figure(figsize =(12,8), dpi=300)
				fig6 = pl.figure(figsize =(12,8), dpi=300)
				fig7 = pl.figure(figsize =(12,8), dpi=300)
				fig8 = pl.figure(figsize =(12,8), dpi=300)
				fig9 = pl.figure(figsize =(12,8), dpi=300)
				fig10 = pl.figure(figsize =(12,8), dpi=300)
				fig11 = pl.figure(figsize =(12,8), dpi=300)
				fig12 = pl.figure(figsize =(12,8), dpi=300)
				fig13 = pl.figure(figsize =(3,2), dpi=300)
				fig14 = pl.figure(figsize =(3,2), dpi=300)
				fig15 = pl.figure(figsize =(12,8), dpi=300)
				fig20 = pl.figure(figsize =(12,8), dpi=300)


				ax = [[] for _ in range(num_timediff)]
				ax2 = [[] for _ in range(num_timediff)]
				ax3 = [[] for _ in range(num_timediff)]
				ax4 = [[] for _ in range(num_timediff)]
				ax5 = [[] for _ in range(num_timediff)]
				ax6 = [[] for _ in range(num_timediff)]
				ax7 = [[] for _ in range(num_timediff)]
				ax8 = [[] for _ in range(num_timediff)]
				ax9 = [[] for _ in range(num_timediff)]
				ax10 = [[] for _ in range(num_timediff)]
				ax11 = [[] for _ in range(num_timediff)]
				ax12 = [[] for _ in range(num_timediff)]
				ax13 = fig13.add_subplot(111)
				ax14 = fig14.add_subplot(111)
				ax15 = [[] for _ in range(num_timediff)]

				
				spikesdiff1 = [[] for _ in range(num_timediff)]
				spikesdiff2 = [[] for _ in range(num_timediff)]

				spikestde = [[] for _ in range(num_timediff)]
				spikestde2 = [[] for _ in range(num_timediff)]
				num_spikestdes = [[] for _ in range(num_timediff)]
				num_spikestdes2 = [[] for _ in range(num_timediff)]
				num_spikestdes_std = [[] for _ in range(num_timediff)]
				num_spikestdes2_std = [[] for _ in range(num_timediff)]
				num_spikesdiff1 = []
				num_spikesdiff2 = []
				num_spikesdiff1_std = []
				num_spikesdiff2_std = []
				num_spikestde = [[] for _ in range(num_timediff)]
				num_spikestde2 = [[] for _ in range(num_timediff)]
				spikestde_all = [[] for _ in range(num_timediff)]
				spikestde2_all = [[] for _ in range(num_timediff)]
				num_spikesdir1 = [[] for _ in range(num_timediff)]
				num_spikesdir2 = [[] for _ in range(num_timediff)]
				num_spikeswta1 = [[] for _ in range(num_timediff)]
				num_spikeswta2 = [[] for _ in range(num_timediff)]
				num_spikesdiffwta = [[] for _ in range(num_timediff)]
				spikeswta1 = [[] for _ in range(num_timediff)]
				spikeswta2 = [[] for _ in range(num_timediff)]
				spikesdiffwta = [[] for _ in range(num_timediff)]
				spikestde_all = [[] for _ in range(num_timediff)]
				spikestde2_all = [[] for _ in range(num_timediff)]
				num_diff_tde = [[] for _ in range(num_timediff)]
				num_diff_tdes = [[] for _ in range(num_timediff)]
				x = 0
				for time_difference in times:
					pathload = ''+str(frequency)+'Hz_itde2/spikes_tde_npy/'
					num_spikestde[x] = np.load(path + 'num_spikestde_fr_'+str(frequency)+neuron_model+'tau_trig'+str(tau_trig)+'pos_'+str(time_difference)+'.npy' )
					num_spikestde2[x] = np.load(path + 'num_spikestde2_fr_'+str(frequency)+neuron_model+'tau_trig'+str(tau_trig)+'pos_'+str(time_difference)+'.npy')
					spikestde[x] = np.load(path + 'spikestde_fr_'+str(frequency)+neuron_model+'tau_trig'+str(tau_trig)+'pos_'+str(time_difference)+'.npy')
					spikestde2[x] = np.load(path + 'spikestde2_fr_'+str(frequency)+neuron_model+'tau_trig'+str(tau_trig)+'pos_'+str(time_difference)+'.npy')
					spikesdiff1[x] = np.load(path + 'spikesdiff1_fr_'+str(frequency)+neuron_model+'tau_trig'+str(tau_trig)+'pos_'+str(time_difference)+'.npy')
					spikesdiff2[x] = np.load(path + 'spikesdiff2_fr_'+str(frequency)+neuron_model+'tau_trig'+str(tau_trig)+'pos_'+str(time_difference)+'.npy')
					num_spikesdiff1.append(np.mean([len(spikesdiff1[x][0]), len(spikesdiff1[x][1]), len(spikesdiff1[x][2])]))
					num_spikesdiff2.append(np.mean([len(spikesdiff2[x][0]), len(spikesdiff2[x][1]), len(spikesdiff2[x][2])]))
					print max(spikesdiff1[x][0])
					print len(spikesdiff1[x][0])
					print len(spikesdiff1[x][1])
					print len(spikesdiff1[x][2])
					num_spikesdiff1_std.append(np.std([len(spikesdiff1[x][0]), len(spikesdiff1[x][1]), len(spikesdiff1[x][2])]))
					num_spikesdiff2_std.append(np.std([len(spikesdiff2[x][0]), len(spikesdiff2[x][1]), len(spikesdiff2[x][2])]))
					num_spikestdes[x] = np.mean(num_spikestde[x], axis = 0)
					num_spikestdes2[x] = np.mean(num_spikestde2[x], axis = 0)
					num_spikestdes_std[x] = np.std(num_spikestde[x], axis = 0)
					num_spikestdes2_std[x] = np.std(num_spikestde2[x], axis = 0)
					x += 1
				

				
				maximum_spikes_tde = float(max(np.amax(num_spikestdes), np.amax(num_spikestdes2)))

				
				# plot tde instantaneous rate
				for position in range(num_timediff):
					for neuron in range(num_tdes):
						spikestde_all[position].append([])
						spikestde2_all[position].append([])
						for channel in range(num_channels):
							spikestde_all[position][neuron] = np.concatenate((spikestde_all[position][neuron], spikestde[position][neuron][channel]), axis=0)
							spikestde2_all[position][neuron] = np.concatenate((spikestde2_all[position][neuron], spikestde2[position][neuron][channel]), axis=0)
			
		

					

					num_diff_tdes[position] = num_spikestdes[position]-num_spikestdes2[position]
				

					
					pl.close('all')
					

					
				mpl.rcParams.update(mpl.rcParamsDefault)
				
				['-90deg, -880us', '-67.5deg, -810us', '-45deg, -610us', '-22.5deg, -320us','0deg, 0us', '22.5deg, 320us', '45deg, 610us', '67.5deg, 810us', '90deg, 880us']
				
				position_angles = [-880, -810, -610, -320, 0, 320, 610, 810, 880]
				time_diff = np.arange(-time_differences, time_differences+res_time_diff, res_time_diff)*100
				timewindow = 4.0
				pl.rc("font", size=5)
				ax13.plot(time_diff, np.array(num_spikesdiff1)/timewindow, c='red')
				ax13.plot(time_diff, np.array(num_spikesdiff2)/timewindow, c='blue')
				ax13.plot(time_diff, np.array(num_spikesdiff1)/timewindow, c='red', marker='.')
				ax13.plot(time_diff, np.array(num_spikesdiff2)/timewindow, c='blue', marker='.')
				ax13.fill_between(time_diff, (np.array(num_spikesdiff1) - np.array(num_spikesdiff1_std))/timewindow, (np.array(num_spikesdiff1) + np.array(num_spikesdiff1_std))/timewindow,linestyle='-', color = 'red', alpha = 0.5)
				ax13.fill_between(time_diff, (np.array(num_spikesdiff2) - np.array(num_spikesdiff2_std))/timewindow, (np.array(num_spikesdiff2) + np.array(num_spikesdiff2_std))/timewindow,linestyle='-', color = 'blue', alpha = 0.5)
				ax13.set_ylabel('frequency (Hz)')
				ax13.set_xlabel('time difference (us)')
				ax13.set_xlim([-900,900])
				
				colors = [ 'midnightblue', 'mediumblue', 'blue', 'dodgerblue', 'maroon', 'brown', 'red', 'salmon']
				pl.rc("font", size=5)
				for i in range(4):
					ax14.plot(time_diff, np.array(num_spikestdes).T[0][i]/timewindow, c=colors[i])
					ax14.plot(time_diff, np.array(num_spikestdes2).T[0][i]/timewindow, c=colors[i+4])
					ax14.plot(time_diff, np.array(num_spikestdes).T[0][i]/timewindow, c=colors[i], marker='*', ms=3.0)
					ax14.plot(time_diff, np.array(num_spikestdes2).T[0][i]/timewindow, c=colors[i+4], marker='o', ms=3.0)
					ax14.fill_between(time_diff, (np.array(num_spikestdes).T[0][i] - np.array(num_spikestdes_std).T[0][i])/timewindow, (np.array(num_spikestdes).T[0][i] + np.array(num_spikestdes_std).T[0][i])/timewindow,linestyle='-', color =colors[i], alpha = 0.5)
					ax14.fill_between(time_diff, (np.array(num_spikestdes2).T[0][i] - np.array(num_spikestdes2_std).T[0][i])/timewindow, (np.array(num_spikestdes2).T[0][i] + np.array(num_spikestdes2_std).T[0][i])/timewindow,linestyle='-', color =colors[i+4], alpha = 0.5)
				
				ax14.plot(time_diff, (np.array(num_spikestdes).T[0][3]+np.array(num_spikestdes2).T[0][3])/timewindow, c='orange', marker='.')
				ax14.plot(time_diff, (np.array(num_spikestdes).T[0][3]+np.array(num_spikestdes2).T[0][3])/timewindow, c='orange')
				ax14.fill_between(time_diff, ((np.array(num_spikestdes).T[0][3]+np.array(num_spikestdes2).T[0][3]) - ((np.array(num_spikestdes_std).T[0][i]+np.array(num_spikestdes2_std).T[0][i])/2))/timewindow, \
				((np.array(num_spikestdes).T[0][3]+np.array(num_spikestdes2).T[0][3]) + ((np.array(num_spikestdes_std).T[0][i]+np.array(num_spikestdes2_std).T[0][i])/2))/timewindow,linestyle='-', color='orange', alpha = 0.5)
				ax14.set_ylabel('frequency (Hz)')
				ax14.set_xlabel('time difference (us)')
				ax14.set_xlim([-900,900])
				
				
				fig13.tight_layout()
				fig13.savefig(pathsave+'/raste_diff_frequ_'+str(frequency)+neuron_model+'tau_trig'+str(tau_trig)+'.png', dpi = 300)
				fig14.tight_layout()
				fig14.savefig(pathsave+'/raste_tde_frequ_'+str(frequency)+neuron_model+'tau_trig'+str(tau_trig)+'.png', dpi = 300)
								
				
				pl.close('all')#			
		
		

    




