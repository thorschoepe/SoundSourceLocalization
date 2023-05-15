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
import re
from scipy.optimize import curve_fit

def func(x, a, b, c):
	return a*x**2 + b*x + c
	
def func2(x, a, b, c):
	return -a*x + b
	 

def binning(spiketimes, simtime, binnumber):
	n = []
	fig = plt.figure(figsize =(8,3.5), dpi=300)
	ax = fig.add_subplot(111)
	for i in range(len(np.array(spiketimes[0].segments[0].spiketrains))):
		data = ax.hist(np.array(spiketimes[0].segments[0].spiketrains[i])/1000.0-10.0, int(binnumber),range=(simtime), density=False, alpha = 1.0, color = 'b')
		n.append(data[0])
	plt.close('all')
	return n
	
def histogram(values, simtime, binnumber):
	n = []
	fig = plt.figure(figsize =(8,3.5), dpi=300)
	ax = fig.add_subplot(111)
	for i in range(len(values)):
		data = ax.hist(values, bins=len(values),range=(simtime[0],simtime[1]), density=False, alpha = 1.0, color = 'b')
		n.append(data[0])
	plt.close('all')
	return n


fig5 = plt.figure(figsize =(4,3), dpi=300)
ax5 = fig5.add_subplot(111)

frequencies = ['250', '500']

for run in range(len(frequencies)):
	path_files = '/home/spinnaker/Documents/projects/Sevilla_2021_TDE_SSL/data/final/closed_loop/'+frequencies[run]+'Hz/'
	files2 = []
	files = []
	for r, d, f in os.walk(path_files):
		for file in f:
			if 'sweep_frequ' in file and '.npy' in file:
				files.append(file)
	for i in range(len(files)):
		files2.append(re.sub('\.npy$', '', files[i]))
				
	files.sort()
	#print files
	
	
	

	fig2 = plt.figure(figsize =(4,3), dpi=300)
	ax2 = fig2.add_subplot(111)
	fig3 = plt.figure(figsize =(4,3), dpi=300)
	ax3 = fig3.add_subplot(111)
	fig4 = plt.figure(figsize =(4,3), dpi=300)
	ax4 = fig4.add_subplot(111)

	
	n_tde = [[] for _ in range(len(files))]
	n_outr = [[] for _ in range(len(files))]
	n_outl = [[] for _ in range(len(files))]
	angles = [[] for _ in range(len(files))]
	#angles = []
	for idx in range(len(files)):
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
		Spikes.p_en_right, Spikes.p_en_left, Spikes.e_pg_right, Spikes.e_pg_left, Spikes.tde_in, Spikes.out_r, Spikes.out_l, Spikes.onset, Spikes.gi = np.load(path_files+files[idx])
		#print 'load angle'
		angle = np.load('/home/spinnaker/Documents/projects/Sevilla_2021_TDE_SSL/data/final/closed_loop/500Hz/500_0'+str(idx+1)+'.mov_angle.npy')
		#print 'done'
		spikedata = [Spikes.tde_in]
		binnumber = 10.0
		simtime = 5.0
		multiplier = binnumber/simtime
		n_tde[idx] = binning([Spikes.tde_in], [0,simtime], binnumber)
		n_outr[idx] = binning([Spikes.out_r], [0,simtime], binnumber)
		n_outl[idx] = binning([Spikes.out_l], [0,simtime], binnumber)
		
		
		maximum = simtime*25
		bins = maximum/binnumber
		#print bins
		for y in range(int(binnumber)):
				angles[idx].append(np.mean(angle[int((y*bins)):int((y*bins+bins))]))
		#print angles
			

			
		


		#ax2.plot((n1[0]-n1[4])*multiplier, (n2[0]-n3[0])*multiplier, '.b')
		#ax2.plot((n1[1]-n1[5])*multiplier, (n2[0]-n3[0])*multiplier, '.r')
		#ax2.plot((n1[2]-n1[6])*multiplier, (n2[0]-n3[0])*multiplier, '.c')
		#ax2.plot((n1[3]-n1[7])*multiplier, (n2[0]-n3[0])*multiplier, '.m')
		#ax2.set_xlabel('tde frequency [Hz]')
		#ax2.set_ylabel('output frequency [Hz]')
		#fig2.savefig('/home/spinnaker/Documents/projects/Sevilla_2021_TDE_SSL/data/final/closed_loop/'+frequencies[run]+'Hz/results/difference'+frequencies[run]+files2[idx]+'Hz.png')
		

			
	
	n_tde_preferred = [[] for _ in range(4)]
	n_out_preferred = []
	n_tde_null = [[] for _ in range(4)]
	n_out_null = []
	
	for i in range(len(files)):
		if i <= 3:
			for z in range(4):
				n_tde_preferred[z] = np.append(n_tde_preferred[z], n_tde[i][z])
			n_out_preferred = np.append(n_out_preferred, n_outr[i])
		elif i >= 5:
			for z in range(4):
				n_tde_preferred[z] = np.append(n_tde_preferred[z], n_tde[i][z+4])
			n_out_preferred = np.append(n_out_preferred, n_outl[i])
	
	popt = [[] for _ in range(4)]
	xsorted = [[] for _ in range(4)]
	r_squared = [[] for _ in range(4)]
	for i in range(4):
		x = [np.array(n_tde_preferred[i][z])*multiplier for z in range(len(n_tde_preferred[1])) if np.array(n_out_preferred[z])*multiplier > 50]
		xsorted[i] = np.sort(x)
		y = [np.array(n_out_preferred[z])*multiplier for z in range(len(n_out_preferred)) if np.array(n_out_preferred[z])*multiplier > 50]
		x = np.array(x)
		y = np.array(y)
		#print x
		
		popt[i], pcov = curve_fit(func, x,y)
		
		
		residuals = y- func(x, *popt[i])
		ss_res = np.sum(residuals**2)
		ss_tot = np.sum((y-np.mean(y))**2)
		r_squared[i] = 1 - (ss_res / ss_tot)
	
	x = np.array([np.array(n_tde_preferred[0][z])*multiplier for z in range(len(n_tde_preferred[0])) if np.array(n_out_preferred[z])*multiplier > 50])
	y = np.array([np.array(n_out_preferred[z])*multiplier for z in range(len(n_out_preferred)) if np.array(n_out_preferred[z])*multiplier > 50])
	popt_linear, pcov = curve_fit(func2, x,y)
	residuals = y- func2(x, *popt_linear)
	ss_res = np.sum(residuals**2)
	ss_tot = np.sum((y-np.mean(y))**2)
	r_squared_linear = 1 - (ss_res / ss_tot)
		
	
	#440, 315, 190 and 63~$\mu$s
	blue_patch = mpatches.Patch(color='maroon', label='tau fac 63us, r2 ='+ str(r_squared[0].round(2)))
	black_patch = mpatches.Patch(color='k', label='linear fit 63us, r2 ='+ str(r_squared_linear.round(2)))
	red_patch = mpatches.Patch(color='brown', label='190us, r2 ='+ str(r_squared[1].round(2)))
	cyan_patch = mpatches.Patch(color='red', label='315us, r2 ='+ str(r_squared[2].round(2)))
	magenta_patch = mpatches.Patch(color='salmon', label='440us, r2 ='+ str(r_squared[3].round(2)))
	colors = [ 'maroon',  'brown', 'red', 'salmon']
	
	#print n_tde_preferred[0]
	#print n_out_preferred
	ax3.plot(np.array(n_tde_preferred[0])*multiplier, np.array(n_out_preferred)*multiplier, marker='.', markerfacecolor='maroon', lw=0.0, c='maroon')
	ax3.plot(np.array(n_tde_preferred[1])*multiplier, np.array(n_out_preferred)*multiplier, marker='+', markerfacecolor='brown', lw=0.0, c='brown')
	ax3.plot(np.array(n_tde_preferred[2])*multiplier, np.array(n_out_preferred)*multiplier, marker='<', markerfacecolor='red', lw=0.0, c='red')
	ax3.plot(np.array(n_tde_preferred[3])*multiplier, np.array(n_out_preferred)*multiplier, marker='*', markerfacecolor='salmon', lw=0.0, c='salmon')
	for i in range(4):
		ax3.plot(xsorted[i], func(xsorted[i], *popt[i]), '-', c =colors[i], label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt[i]))
	ax3.plot(xsorted[0], func2(xsorted[0], *popt_linear), '-', c ='k', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt_linear))
	ax3.legend(handles=[blue_patch, black_patch, red_patch, cyan_patch, magenta_patch], fontsize=8,loc='lower right' )
	ax3.set_xlabel('tde frequency [Hz]')
	ax3.set_ylabel('output frequency [Hz]')
	fig3.tight_layout()
	fig3.savefig('/home/spinnaker/Documents/projects/Sevilla_2021_TDE_SSL/data/final/closed_loop/'+frequencies[run]+'Hz/results/single_tdes'+frequencies[run]+'Hz.png')
	

	ax4.plot(np.squeeze(angles[0:len(n_out_preferred)]), np.squeeze((np.array(n_outl)))*multiplier, '.b')
	ax4.plot(np.squeeze(angles[0:len(n_out_preferred)]), np.squeeze((np.array(n_outr)))*multiplier, '.r')
	#for i in range(4):
	#	ax3.plot(xsorted[i], func(xsorted[i], *popt[i]), '-', c =colors[i], label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt[i]))
	#ax3.plot(xsorted[0], func2(xsorted[0], *popt_linear), '-', c ='k', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt_linear))
	ax4.legend(handles=[blue_patch, black_patch, red_patch, cyan_patch, magenta_patch], fontsize='x-small')
	ax4.set_xlabel('angle [degrees]')
	ax4.set_ylabel('output frequency [Hz]')
	fig4.tight_layout()
	fig4.savefig('/home/spinnaker/Documents/projects/Sevilla_2021_TDE_SSL/data/final/closed_loop/'+frequencies[run]+'Hz/results/angle_out'+frequencies[run]+'Hz.png')
	plt.close('all')

	blue_patch = mpatches.Patch(color='maroon', label='63us')
	red_patch = mpatches.Patch(color='brown', label='190us')
	cyan_patch = mpatches.Patch(color='red', label='315us')
	magenta_patch = mpatches.Patch(color='salmon', label='440us')
	

	n_outrl = ((np.squeeze(np.array(n_outr))-np.squeeze(np.array(n_outl)))*multiplier).ravel()
	n_anglepref = (np.degrees(np.squeeze(angles[0:len(n_out_preferred)]))-175).ravel()
	
	print n_outrl
	print n_anglepref
	
	xsorted2 = np.sort(n_anglepref)
	

	
	popt_lin, pcov = curve_fit(func2, n_anglepref, n_outrl)
	residuals = y- func2(x, *popt_linear) 
	ss_res = np.sum(residuals**2)
	ss_tot = np.sum((y-np.mean(y))**2)
	r_squared_lin = 1 - (ss_res / ss_tot)
	
	
	if run == 0:
		print xsorted2
		print popt_lin
		ax5.plot(n_anglepref, n_outrl, marker='.', markerfacecolor='darkgreen', lw=0.0, c='darkgreen')
		ax5.plot(xsorted2, func2(xsorted2, *popt_lin), '-', c ='darkgreen', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt_lin))
		darkgreen_patch = mpatches.Patch(color='darkgreen', label='250 Hz, r2='+ str(r_squared_lin.round(2)))
	elif run == 1:
		ax5.plot(n_anglepref, n_outrl, marker='.', markerfacecolor='limegreen', lw=0.0, c='limegreen')
		ax5.plot(xsorted2, func2(xsorted2, *popt_lin), '-', c ='limegreen', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt_lin))
		limegreen_patch = mpatches.Patch(color='limegreen', label='500 Hz, r2='+ str(r_squared_lin.round(2)))
		ax5.legend(handles=[darkgreen_patch, limegreen_patch], fontsize=10)
		ax5.set_xlabel('angle (degrees)')
		ax5.set_ylabel('output frequency right-left (Hz)')
		ax5.set_xlim([-90,90])
	fig5.tight_layout()
	fig5.savefig('/home/spinnaker/Documents/projects/Sevilla_2021_TDE_SSL/data/final/closed_loop/'+frequencies[run]+'Hz/results/single_tdes_out_250_500Hz.png')
	plt.close('all')
		

		
	

		
