import numpy as np
import matplotlib.pyplot as plt
import math
import os, fnmatch
import scipy
import random




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
	tde_in=0
	out_r=0
	out_l=0 



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
	files = find('out_spikes_recording_'+str(recording+1)+'*', '../data/development/spikes/')
	for repetitions in range(5):
		
		
		# load spikes and ground truth data
		Spikes.out_r, Spikes.out_l, Spikes.tde_in = np.load(files[repetitions])
		path = '../data/development/recording'+str(recording+1)+'/dummy/'
		file = open(path +'position_source_talker.txt', 'r')
		file2 = open(path +'position_array_dummy.txt', 'r')
		file3 = open(path +'VAD_dummy_talker.txt', 'r')
		
		# extract matrix from txt file
		data_source = []
		for row in file:
			data_source.append([x for x in row.split()])
		data_source = np.array(data_source[1:]).T

		data_receiv = []
		for row in file2:
			data_receiv.append([x for x in row.split()])
		data_receiv = np.array(data_receiv[1:]).T

		data_voicead = []
		for row in file3:
			data_voicead.append([x for x in row.split()])
		data_voicead = np.array(data_voicead[1:]).flatten()
		
		
		# create color array for voice active and voice inactive time
		colors = []
		ratio = int(len(data_voicead)/len(data_source[0])) # scale time
		for i in range(len(data_source[0])):
			if int(data_voicead[i*ratio]) == 1:
				colors.append('red')
			else:
				colors.append('gray')


		# extract relevant variables from matrix
		data_source = data_source[6:8] # position of source
		data_angle_receiv = [data_receiv[21:23][0][0], data_receiv[21:23][1][0]] # position of microphone 1
		data_angle_receiv_2 = [data_receiv[27:29][0][0], data_receiv[27:29][1][0]] # position of microphone 3
		time_receiv = data_receiv[5].astype(float)-min(data_receiv[5].astype(float)) # time of receiver
		data_receiv = [data_receiv[6:8][0][0], data_receiv[6:8][1][0]] # center position o receiver

		# timing information needs to be corrected if minute changed during recording.
		# Since every audio recording is about 25 seconds long no time larger than 30 seconds should occur, only when minute changed
		# in that case move larger times at beginning
		for i in range(len(time_receiv)):
			if float(time_receiv[i]) > 30.0:
				time_receiv[i] = time_receiv[i]-60
		if min(time_receiv) < 0.0:
			time_receiv = time_receiv - min(time_receiv)
				
		# estimate azimuth angle of receiver
		angle_receiver = math.atan2(float(data_angle_receiv_2[1]) - float(data_angle_receiv[1]), float(data_angle_receiv_2[0]) - float(data_angle_receiv[0])) * 180 / math.pi



		# estimate angle between source and receiver, the precision of this estimate can probably be improved
		angle = []
		for i in range(len(data_source[0])):
			angle.append(math.atan2(float(data_source[1][i]) - float(data_receiv[1]), float(data_source[0][i]) - float(data_receiv[0])) * 180 / math.pi)
			if angle[i] > 0:
				angle[i] = angle[i] - 180
			else:
				angle[i] = angle[i] + 180 





		# estimate instantaneous rate of left and right output spikes
		n = np.zeros(maxtime/step) # instantaneous rate right
		n2 = np.zeros(maxtime/step) # instantaneous rate left
		for i in range(0,maxtime, step):
			for y in range(len(Spikes.out_r)):
				if Spikes.out_r[y] > i-binsize/2.0 and Spikes.out_r[y] < i+ binsize/2.0:
					n[i/step] = n[i/step] + 1
			for y in range(len(Spikes.out_l)):
				if Spikes.out_l[y] > i-binsize/2.0 and Spikes.out_l[y] < i+ binsize/2.0:
					n2[i/step] = n2[i/step] + 1
		

		# Exactly at second 30 in the audacity recording, I play a 500 Hz pure tone with very high amplitude.
		# This pure tone increases the TDE activity immensly
		# here I measure the onset of the increase to calculate the delay between ground truth and played sound
		# Later I add this delay to the time of the ground truth to synchonize the data
		fig1 = plt.figure()
		ax1 = fig1.add_subplot(111)
		n3, bins3, patches3 = ax1.hist(Spikes.tde_in,50,range = [30000,35000], normed=False, alpha = 0.5, color = 'k') # if parameter normed unknown use density instead
		for i in range(len(n3)):
			if n3[i+1] > 20:
				delay = i*100
				break
		time_receiv = time_receiv*1000+delay # add delay to ground truth and move to ms range
		x_value = range(0,maxtime, step)
		
		# select all instantaneous rate data during voice active time, the arrays with ending "a" collect all data for all runs
		spikes_talking=[]
		time_talking =[] 
		angle_talking = []
		time_receiv_talking = []
		for i in range(len(colors)):
			if colors[i] == 'red':
				for y in range(len(x_value)):
					if x_value[y]- time_receiv[i] > 0:
						index = y
						break
				spikes_talking.append(n[index]-n2[index])
				time_talking.append(x_value[index])
				angle_talking.append(angle[i])
				time_receiv_talking.append(time_receiv[i])
				spikes_talking_a.append(n[index]-n2[index])
				time_talking_a.append(x_value[index])
				angle_talking_a.append(angle[i])
				time_receiv_talking_a.append(time_receiv[i])

					
		# sort angle and instantaneous rate 
		zipped_lists = zip(angle_talking_a,spikes_talking_a)
		sorted_pairs = sorted(zipped_lists)
		tuples = zip(*sorted_pairs)
		angle_talking_a, spikes_talking_a = [ list(tuple) for tuple in  tuples]
		

					


		
		# plot rate and angle
		fig2 = plt.figure(figsize =(6,8), dpi=300)
		plt.rc("font", size=5)
		ax2 = fig2.add_subplot(2,1,1)
		ax2.plot(time_talking, spikes_talking, '.r')
		ax2.set_xlim([0,maxtime])
		ax2.set_ylabel('out (rate)')
		ax2.set_xlabel('time (ms)')
		ax4 = fig2.add_subplot(2,1,2)
		for i in range(len(angle)):
			ax4.plot(time_receiv[i], angle[i],  marker ='.', color = colors[i])
		ax4.set_xlim([0,maxtime])
		ax4.set_ylabel('azimuth angle (deg)')
		ax4.set_xlabel('time (ms)')


		# plot rate over angle
		heatmap, xedges, yedges = np.histogram2d(angle_talking, spikes_talking, bins=11)
		extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
		fig3 = plt.figure(figsize =(6,4), dpi=300)
		plt.rc("font", size=5)
		ax3 = fig3.add_subplot(1,1,1)
		ax3.imshow(heatmap.T, extent=extent, origin='lower')
		ax3.set_ylabel('delta rate')
		ax3.set_xlabel('angle')
		ax3.set_aspect(0.1)
		
		# plot rate over angle all together
		heatmap, xedges, yedges = np.histogram2d(angle_talking_a, spikes_talking_a, bins=11)
		heatmap = heatmap.T
		extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
		fig4 = plt.figure(figsize =(6,4), dpi=300)
		plt.rc("font", size=5)
		ax3 = fig4.add_subplot(121)
		ax3.imshow(heatmap, extent=extent, origin='lower')
		ax3.set_ylabel('delta rate')
		ax3.set_xlabel('angle')
		ax3.set_aspect(0.1)
		ax4 = fig4.add_subplot(122)
		ax4.plot(angle_talking_a, spikes_talking_a, '.b')
			
		
		# estimat error between rate and angle, transfer function missing at the moment
		error.append(sum(abs(np.array(spikes_talking_a)-np.array(angle_talking_a)))/len(spikes_talking_a))
		print 'error:'
		print error


		
		fig4.savefig('../data/development/heatmap.png', dpi=300)
		fig3.savefig('../data/development/heatmap' +str(recording+1)+'_' + str(repetitions+1) +'.png', dpi=300)
		fig2.savefig('../data/development/recording'+str(recording+1)+'_' + str(repetitions+1) +'.png', dpi=300)


		plt.close('all')

print 'average error:'
print sum(np.array(error))/float(len(error))
		


