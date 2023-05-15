import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import spynnaker8 as p 
from matplotlib import gridspec
from quantities import ms
import os


	
def histogram_overlay_mean(spiketimes,error,populations_size, simtime, x_labels, y_labels, path_save, x_lim, color):

	number_subplots = len(spiketimes)
	ax = [[] for _ in range(number_subplots+1)]


	fig = plt.figure(figsize =(3,2), dpi=300)
	plt.rc("font", size=5)
	ax = fig.add_subplot(111)

	

	for num in range(1, number_subplots+1):
		ax.set_ylabel(y_labels[num-1])
		ax.set_xlabel(x_labels)
		ax.set_xlim(x_lim)
		ax.plot(range(len(spiketimes[num-1])), spiketimes[num-1], marker='.', c = color[num-1], alpha = 0.5)
		ax.fill_between(range(len(spiketimes[num-1])), np.array(spiketimes[num-1]) - np.array(error[num-1]), np.array(spiketimes[num-1]) + np.array(error[num-1]),linestyle='None', color = color[num-1], alpha = 0.5)
		ax.plot(range(len(spiketimes[num-1])), spiketimes[num-1], marker = '.', c = color[num-1], alpha = 0.5)
			
	fig.tight_layout()
		
	fig.savefig(path_save)
	plt.close('all')
	
def histogram_overlay(spiketimes,populations_size, simtime, x_labels, y_labels, path_save, x_lim):

	number_subplots = len(spiketimes)
	ax = [[] for _ in range(number_subplots+1)]


	fig = plt.figure(figsize =(6,4), dpi=300)
	plt.rc("font", size=5)
	ax = fig.add_subplot(111)

		

	for num in range(1, number_subplots+1):
		spikes = []
		for i in range(len(spiketimes[num-1])):
			spikes = np.append(spikes, spiketimes[num-1][i]/ms)
		ax.set_ylabel(y_labels[num-1])
		ax.set_xlabel(x_labels)
		ax.set_xlim(x_lim)
		ax.hist(np.array(spikes).T.flatten(), 40, density=False, alpha = 0.5)
			
	fig.tight_layout()
		
	fig.savefig(path_save)
	plt.close('all')
	
def histogram_overlay_mean_angle(spiketimes,error,populations_size, simtime, x_labels, y_labels, path_save, x_lim, color):

	number_subplots = len(spiketimes)
	ax = [[] for _ in range(number_subplots+1)]


	fig = plt.figure(figsize =(3,2), dpi=300)
	plt.rc("font", size=5)
	ax = fig.add_subplot(111)

	

	for num in range(1, number_subplots+1):
		ax.set_ylabel(y_labels[num-1])
		ax.set_xlabel(x_labels)
		ax.set_xticks(np.arange(-90.0, 95, step=22.5)) 
		#ax.set_xlim(x_lim)
		ax.plot(np.arange(len(spiketimes[num-1]))*5.625-92.0-11.25-22.5, spiketimes[num-1], marker='.', c = color[num-1], alpha = 0.5)
		ax.fill_between(np.arange(len(spiketimes[num-1]))*5.625-92.0-11.25-22.5, np.array(spiketimes[num-1]) - np.array(error[num-1]), np.array(spiketimes[num-1]) + np.array(error[num-1]),linestyle='None', color = color[num-1], alpha = 0.5)
		ax.plot(np.arange(len(spiketimes[num-1]))*5.625-92.0-11.25-22.5, spiketimes[num-1], marker = '.', c = color[num-1], alpha = 0.5)
		ax.set_xlim([-95,95])
	fig.tight_layout()
		
	fig.savefig(path_save)
	plt.close('all')
	

def histogram_values(spiketimes):
		spikes = []
		print len(spiketimes)
		for i in range(len(spiketimes)):
			spikes = np.append(spikes, spiketimes[i]/ms)
		
		(n, bins, patches) = plt.hist(np.array(spikes).T.flatten(), 40, density=False)

		return n




simtime = 40000





channels = ['onechannel', 'multichannel']
	
for frequency in range(250,1500,250):
	for channeltype in range(len(channels)):
		
		hist_val_out_r = []
		hist_val_out_l = []
		hist_val_tde_r = []
		hist_val_tde_l = []
		hist_val_tdes_r = []
		hist_val_tdes_l = []
		std_tdes_r = []
		std_tdes_l = []
		mean_tdes_r = []
		mean_tdes_l = []
		hist_val_tdes_r1 = []
		hist_val_tdes_l1 = []
		std_tdes_r1 = []
		std_tdes_l1 = []
		mean_tdes_r1 = []
		mean_tdes_l1 = []
		hist_val_tdes_r2 = []
		hist_val_tdes_l2 = []
		std_tdes_r2 = []
		std_tdes_l2 = []
		mean_tdes_r2 = []
		mean_tdes_l2 = []
		hist_val_tdes_r3 = []
		hist_val_tdes_l3 = []
		std_tdes_r3 = []
		std_tdes_l3 = []
		mean_tdes_r3 = []
		mean_tdes_l3 = []
		hist_val_tdes_rl3 = []
		std_tdes_rl3 = []
		mean_tdes_rl3 = []

		
		class Spikes:
			pass	
			p_en_right=0
			p_en_left=0
			e_pg_right=0
			e_pg_left=0
			tde_in=0
			onset=0
			tde_in_right = 0
			tde_in_left = 0
			out_r = 0
			out_l = 0

		path = '/home/spinnaker/Documents/projects/Sevilla_2021_TDE_SSL/data/final/sweep/'+channels[channeltype] +'/' + str(frequency) + 'Hz/'
		fileExt = r".npy"
		files = [_ for _ in os.listdir(path) if _.endswith(fileExt)]
		for fil in range(len(files)):
			
			Spikes.p_en_right, Spikes.p_en_left, Spikes.e_pg_right, Spikes.e_pg_left, Spikes.tde_in, Spikes.out_r, Spikes.out_l, Spikes.onset = np.load(path + files[fil])
			
			
			### plots ########################################################################################################################################################
			
			
			for i in range(4):
				Spikes.tde_in_right = np.append(Spikes.tde_in_right, Spikes.tde_in.segments[0].spiketrains[i])
				Spikes.tde_in_left = np.append(Spikes.tde_in_left, Spikes.tde_in.segments[0].spiketrains[i+4])
				
			
			#Spikes.tde_in.append(tde_in.segments[0].spiketrains)
			#Spikes.out_r.append(out_r.segments[0].spiketrains)
			#Spikes.out_l.append(out_l.segments[0].spiketrains)
			#Spikes.wta_r.append(wta_r.segments[0].spiketrains)
			#Spikes.wta_l.append(wta_l.segments[0].spiketrains)

			
			hist_val_out_r.append(histogram_values(Spikes.out_r.segments[0].spiketrains))
			hist_val_out_l.append(histogram_values(Spikes.out_l.segments[0].spiketrains))
			std_out_r = np.std(hist_val_out_r, axis=0)
			std_out_l = np.std(hist_val_out_l, axis=0)
			mean_out_r = np.mean(hist_val_out_r, axis=0)
			mean_out_l = np.mean(hist_val_out_l, axis=0)

			hist_val_tde_r.append(histogram_values(Spikes.tde_in_right))
			hist_val_tde_l.append(histogram_values(Spikes.tde_in_left))
			std_tde_r = np.std(hist_val_tde_r, axis=0)
			std_tde_l = np.std(hist_val_tde_l, axis=0)
			mean_tde_r = np.mean(hist_val_tde_r, axis=0)
			mean_tde_l = np.mean(hist_val_tde_l, axis=0)
			

			

			hist_val_tdes_r.append(histogram_values(Spikes.tde_in.segments[0].spiketrains[0]))
			hist_val_tdes_l.append(histogram_values(Spikes.tde_in.segments[0].spiketrains[4]))
			std_tdes_r = np.std(hist_val_tdes_r, axis=0)
			std_tdes_l = np.std(hist_val_tdes_l, axis=0)
			mean_tdes_r = np.mean(hist_val_tdes_r, axis=0)
			mean_tdes_l = np.mean(hist_val_tdes_l, axis=0)
			
			hist_val_tdes_r1.append(histogram_values(Spikes.tde_in.segments[0].spiketrains[1]))
			hist_val_tdes_l1.append(histogram_values(Spikes.tde_in.segments[0].spiketrains[5]))
			std_tdes_r1 = np.std(hist_val_tdes_r1, axis=0)
			std_tdes_l1 = np.std(hist_val_tdes_l1, axis=0)
			mean_tdes_r1 = np.mean(hist_val_tdes_r1, axis=0)
			mean_tdes_l1 = np.mean(hist_val_tdes_l1, axis=0)
			
			hist_val_tdes_r2.append(histogram_values(Spikes.tde_in.segments[0].spiketrains[2]))
			hist_val_tdes_l2.append(histogram_values(Spikes.tde_in.segments[0].spiketrains[6]))
			std_tdes_r2 = np.std(hist_val_tdes_r2, axis=0)
			std_tdes_l2 = np.std(hist_val_tdes_l2, axis=0)
			mean_tdes_r2 = np.mean(hist_val_tdes_r2, axis=0)
			mean_tdes_l2 = np.mean(hist_val_tdes_l2, axis=0)
			
			hist_val_tdes_r3.append(histogram_values(Spikes.tde_in.segments[0].spiketrains[3]))
			hist_val_tdes_l3.append(histogram_values(Spikes.tde_in.segments[0].spiketrains[7]))
			std_tdes_r3 = np.std(hist_val_tdes_r3, axis=0)
			std_tdes_l3 = np.std(hist_val_tdes_l3, axis=0)
			mean_tdes_r3 = np.mean(hist_val_tdes_r3, axis=0)
			mean_tdes_l3 = np.mean(hist_val_tdes_l3, axis=0)

			hist_val_tdes_rl3.append(histogram_values(np.append(Spikes.tde_in.segments[0].spiketrains[3], Spikes.tde_in.segments[0].spiketrains[7])))
			std_tdes_rl3 = np.std(hist_val_tdes_rl3, axis=0)
			mean_tdes_rl3 = np.mean(hist_val_tdes_rl3, axis=0)

				

		print mean_tdes_r[0]
			
		#print mean_tdes_l
			
		color = [ 'midnightblue', 'maroon', 'mediumblue', 'brown', 'blue', 'red', 'dodgerblue',  'salmon', 'orange']
		y_labels_hist = [('frequency [Hz]') for i in range(9)]
		x_lim = [0,37]
		path_hist_mean = path + str(frequency) + 'Hz_'+str(channeltype)+'mean_SSL_tdes2.png'
		histogram_overlay_mean([mean_tdes_r, mean_tdes_l, mean_tdes_r1, mean_tdes_l1, mean_tdes_r2, mean_tdes_l2, mean_tdes_r3, mean_tdes_l3, mean_tdes_rl3],[std_tdes_r, std_tdes_l,std_tdes_r1, std_tdes_l1,std_tdes_r2, std_tdes_l2,std_tdes_r3, std_tdes_l3, std_tdes_rl3],[1,1, 1, 1], simtime,'time (sec)', y_labels_hist, path_hist_mean, x_lim, color)
		
		color = ['b', 'r']
		y_labels_hist_2 = ['frequency [Hz]', 'frequency [Hz]']
		x_lim = [0,37]
		path_hist_mean = path + str(frequency) +'Hz_'+str(channeltype)+'mean_SSL_out2.png'
		histogram_overlay_mean([mean_out_r, mean_out_l],[std_out_r, std_out_l],[1,1], simtime,'time (sec)', y_labels_hist_2, path_hist_mean, x_lim, color)
		
		color = ['b', 'r']
		y_labels_hist_2 = ['frequency [Hz]', 'frequency [Hz]']
		x_lim = [0,37]
		path_hist_mean = path + str(frequency) +'Hz_'+str(channeltype)+'mean_SSL_tde2.png'
		histogram_overlay_mean([mean_tde_r, mean_tde_l],[std_tde_r, std_tde_l],[1,1], simtime,'time (sec)', y_labels_hist_2, path_hist_mean, x_lim, color)

	



	







