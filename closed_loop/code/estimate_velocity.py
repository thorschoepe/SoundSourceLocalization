import numpy as np
import cv2
import time
import glob
import os
import argparse
from scipy.spatial import distance
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches

colors = ['darkorange', 'red', 'blue', 'darkorchid', 'hotpink', 'cornflowerblue', 'maroon', 'fuchsia', 'chocolate']


def angle_between(p1, p2):
	angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0]) * (180 / math.pi)
	return math.radians(angle)

frequency = [500]
angle_mean = [[] for _ in range(len(frequency))]
angle_std = [[] for _ in range(len(frequency))]

blue_patch = mpatches.Patch(color='blue', label='250 Hz')
red_patch = mpatches.Patch(color='red', label='500 Hz')
magenta_patch = mpatches.Patch(color='magenta', label='speech')

patch1 = mpatches.Patch(color=colors[0], label='5 sec')
patch2 = mpatches.Patch(color=colors[1], label='4 sec')
patch3 = mpatches.Patch(color=colors[2], label='3 sec')
patch4 = mpatches.Patch(color=colors[3], label='2 sec')
patch5 = mpatches.Patch(color=colors[4], label='1 sec')

lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])





for frequ in range(len(frequency)):
	
	fig2 = plt.figure(figsize =(4,4), dpi=300)
	plt.rc("font", size=10)
	ax2 = fig2.add_subplot(111, projection = 'polar')

	fig3 = plt.figure(figsize =(4,4), dpi=300)
	plt.rc("font", size=10)
	ax3 = fig3.add_subplot(111, projection = 'polar')
	
	fig4 = plt.figure(figsize =(8,3), dpi=300)
	plt.rc("font", size=10)
	ax4 = fig4.add_subplot(111)

	
	path = '/home/spinnaker/Documents/projects/Sevilla_2021_TDE_SSL/data/final/closed_loop/'+str(frequency[frequ])+'Hz/'
	#video_name = []
	#os.chdir(path)
	#for file in glob.glob("*.webm"):
	#	video_name.append(file)
	video_name = ['sweep_frequ500_datetime_22_02_2022_18-27-25.webm']
	picture_name = video_name
	angle_all = [[] for _ in range(len(video_name))]
	#angle_all = [[] for _ in range(2)]
	velocity_all = [[] for _ in range(len(video_name))]
	velocity_all_2 = [[] for _ in range(len(video_name))]
	velocity_all_3 = [[] for _ in range(len(video_name))]
	velocity_all_4 = [[] for _ in range(len(video_name))]
	velocity_all_5 = [[] for _ in range(len(video_name))]
	
	angle_start = [[] for _ in range(len(video_name))]

	
	for idx in range(len(video_name)):
	#for idx in range(2):
		angle = []
		cap = cv2.VideoCapture(path+video_name[idx])
		# 25 fps
		print video_name[idx]
		
		
		
		while True:
			ret, frame = cap.read()
			
			
			
			if frame == None:
				break
			
			frame = frame[0:719, 300:1279]
			hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
			mask = cv2.inRange(hsv, lower_red, upper_red)

			# Bitwise-AND mask and original image
			frame_red = cv2.bitwise_and(frame,frame, mask= mask)
			
			
			  


			# converting image into grayscale image
			gray = cv2.cvtColor(frame_red, cv2.COLOR_BGR2GRAY)
			  
			# setting threshold of gray image
			_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
			
			kernel = np.ones((5,5),np.float32)/25
			threshold = cv2.dilate(threshold,kernel,iterations = 5)
			

			# using a findContours() function
			contours, _ = cv2.findContours(
				threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			# find biggest contour	
			sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
			largest_item= sorted_contours[0]
			
			

			# cv2.approxPloyDP() function to approximate the shape
			approx = cv2.approxPolyDP(largest_item, 0.01 * cv2.arcLength(largest_item, True), True)
			rect = cv2.minAreaRect(largest_item)
			# box includes four corner points of rectangle object
			box = cv2.cv.BoxPoints(rect)
			#print box[0]
			
			#cv2.rectangle(frame_red, (int(box[0][0]),int(box[0][1]) ), (int(box[2][0]),int(box[2][1])), (255,0,0), 2)
			#cv2.imshow('mask',frame_red)
			#cv2.waitKey(0)
			

			if video_name[idx] == str(frequency[frequ])+'_09.mov':
				if distance.euclidean(box[0], box[1]) > distance.euclidean(box[1], box[2]):
					angle_value = angle_between(box[0], box[1])
					if angle_value < 0 and angle_value > -1.8:
						angle.append(angle_value+math.pi)
					else:
						angle.append(angle_value)
				else:
					angle_value = angle_between(box[2], box[1])
					if angle_value < 0 and angle_value > -1.8:
						angle.append(angle_value+math.pi)
					else:
						angle.append(angle_value)
			
			else:
				if distance.euclidean(box[0], box[1]) > distance.euclidean(box[1], box[2]):
					angle.append(angle_between(box[0], box[1]))
				else:
					angle.append(angle_between(box[2], box[1]))
					
					
		angle_all[idx] = angle
		
		print np.degrees(angle)
		
		for i in range(len(angle)):
			if np.degrees(angle[i]) <= 0:
				angle[i] = angle[i] + 2*np.pi 
		
		
		velocity_all[idx] = np.diff(np.degrees(angle[0:124]))*25.0
		velocity_all_2[idx] = np.diff(np.degrees(angle[0:99]))*25.0
		velocity_all_3[idx] = np.diff(np.degrees(angle[0:74]))*25.0
		velocity_all_4[idx] = np.diff(np.degrees(angle[0:49]))*25.0
		velocity_all_5[idx] = np.diff(np.degrees(angle[0:24]))*25.0
		
		threshold = 200
		for i in range(1,len(velocity_all[idx]),1):
			if abs(velocity_all[idx][i]) > threshold: 
				velocity_all[idx][i] = velocity_all[idx][i-1]
		for i in range(1,len(velocity_all_2[idx]),1):
			if abs(velocity_all_2[idx][i]) > threshold: 
				velocity_all_2[idx][i] = velocity_all_2[idx][i-1]
		for i in range(1,len(velocity_all_3[idx]),1):
			if abs(velocity_all_3[idx][i]) > threshold: 
				velocity_all_3[idx][i] = velocity_all_3[idx][i-1]
		for i in range(1,len(velocity_all_4[idx]),1):
			if abs(velocity_all_4[idx][i]) > threshold: 
				velocity_all_4[idx][i] = velocity_all_4[idx][i-1]
		for i in range(1,len(velocity_all_4[idx]),1):
			if abs(velocity_all_4[idx][i]) > threshold: 
				velocity_all_4[idx][i] = velocity_all_4[idx][i-1]
		for i in range(1,len(velocity_all_5[idx]),1):
			if abs(velocity_all_5[idx][i]) > threshold: 
				velocity_all_5[idx][i] = velocity_all_5[idx][i-1]
				
		velocity_all[idx] = np.mean(velocity_all[idx])
		velocity_all_2[idx] = np.mean(velocity_all_2[idx])
		velocity_all_3[idx] = np.mean(velocity_all_3[idx])
		velocity_all_4[idx] = np.mean(velocity_all_4[idx])
		velocity_all_5[idx] = np.mean(velocity_all_5[idx])
		
		if np.degrees(angle[0]) > 0:
			angle_start[idx] = np.degrees(angle[0])-180
		else:
			angle_start[idx] = np.degrees(angle[0])+180
			
		
		for i in range(len(angle)-2):
			if video_name[idx] != '500_02.mov':
				if abs(np.degrees(abs(angle[i] - angle[i+1])) - np.degrees(abs(angle[i+1] - angle[i+2]))) > 5:
					angle[i+2] = angle[i+1]
			elif video_name[idx] == '500_02.mov':
				print np.degrees(angle[i+1])
				if abs(np.degrees(angle[i+1])) > 179 and abs(np.degrees(angle[i+1])) < 181:
					angle[i+1] = angle[i]
				if abs(np.degrees(abs(angle[i] - angle[i+1])) - np.degrees(abs(angle[i+1] - angle[i+2]))) > 10:
					angle[i+2] = angle[i+1]
				
		#np.save('/home/spinnaker/Documents/projects/Sevilla_2021_TDE_SSL/data/final/closed_loop/'+str(frequency[frequ])+'Hz/' + str(video_name[idx]) + '_angle.npy',angle)

				
		fig = plt.figure()
		plt.rc("font", size=5)
		ax = fig.add_subplot(111, projection = 'polar')
		ax.plot(np.array(angle), np.array(range(len(angle)))/15.0, marker = '.', ms = 1, color = 'b', linewidth = 0)
		ax.set_xticks(np.pi/180. * np.linspace(0,  360, 36, endpoint=False))
		ax.set_ylim([-20,110])
		fig.savefig('/home/spinnaker/Documents/projects/Sevilla_2021_TDE_SSL/data/final/closed_loop/'+str(frequency[frequ])+'Hz/results/' + str(video_name[idx]) + '.png', dpi=300)

		
		ax2.plot(np.array(angle)+np.pi, np.array(range(len(angle)))/15.0, marker = '.', ms = 3, linewidth = 0.5, color= colors[idx])
		ax2.set_xticks(np.pi/180. * np.linspace(0,  360, 36, endpoint=False))
		ax2.set_ylim([-20,110])
		fig2.savefig('/home/spinnaker/Documents/projects/Sevilla_2021_TDE_SSL/data/final/closed_loop/'+str(frequency[frequ])+'Hz/results/polarplot_all_'+str(frequency[frequ])+'uncropped_Hz.png', dpi=300)
		
		ax3.plot(np.array(angle)+np.pi/2, np.array(range(len(angle)))/15.0, marker = '.', ms = 3, linewidth = 0.5, color= colors[idx])
		ax3.set_xticks(np.pi/180. * np.linspace(0,  360, 36, endpoint=False))
		ax3.set_ylim([-10,20])
		fig3.savefig('/home/spinnaker/Documents/projects/Sevilla_2021_TDE_SSL/data/final/closed_loop/'+str(frequency[frequ])+'Hz/results/polarplot_all_turn_'+str(frequency[frequ])+'uncropped_Hz.png', dpi=300)
		
		N = 7.0
		rolling_mean = np.convolve(np.diff(np.array(angle)), np.ones(N)/N, mode='valid')
		plt.rc("font", size=10)
		ax4.plot((np.array(range(len(rolling_mean)))/15.0)-29.0, np.rad2deg(rolling_mean)*15.0, marker = '.', ms = 5, linewidth = 1.0, color= colors[idx])
		ax4.set_ylim([-45,35])
		ax4.set_xlim([-12,6])
		ax4.set_xlabel('time (sec)')
		ax4.set_ylabel('velocity (deg/sec)')
		ax4.set_xticks(range(-12,8,4))
		ax4.set_yticks(range(-40,40,10))
		ax4.grid(which='major', color='#DDDDDD', linewidth=1.0)
		ax4.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.7)
		ax4.minorticks_on()
		fig4.tight_layout()
		fig4.savefig('/home/spinnaker/Documents/projects/Sevilla_2021_TDE_SSL/data/final/closed_loop/'+str(frequency[frequ])+'Hz/results/velocity_'+str(frequency[frequ])+'uncropped_Hz.png', dpi=300)
		
		
	plt.close('all')


	len_all = []
	for i in range(len(angle_all)):
		len_all.append(len(angle_all[i]))

	for i in range(len(angle_all)):
		angle_all[i] = angle_all[i][:min(len_all)-1]
		


	for i in range(len(angle_all[0])):
		#print np.array(angle_all).T[i]%(2*np.pi)
		#print np.mean(np.array(angle_all).T[i]%(2*np.pi))
		angle_std[frequ].append(np.std(np.array(angle_all).T[i]%(2*np.pi)))
		angle_mean[frequ].append(np.mean(np.array(angle_all).T[i]%(2*np.pi)))
		
	zipped_lists = zip(angle_start,velocity_all,velocity_all_2, velocity_all_3, velocity_all_4, velocity_all_5)
	sorted_pairs = sorted(zipped_lists)
	tuples = zip(*sorted_pairs)
	angle_start,velocity_all,velocity_all_2, velocity_all_3, velocity_all_4, velocity_all_5 = [ list(tuple) for tuple in  tuples]





fig = plt.figure(figsize =(4,4), dpi=300)
plt.rc("font", size=7)
ax = fig.add_subplot(111, projection = 'polar')
ax.plot(np.array(angle_mean[0])+np.pi, np.array(range(len(angle_mean[0])))/25.0, marker = '.', ms = 2, linewidth = 0, color = 'b')
ax.fill_betweenx(np.array(range(len(angle_mean[0])))/25.0,np.array(angle_mean[0])-np.array(angle_std[0])+np.pi, np.array(angle_mean[0])+np.array(angle_std[0])+np.pi, color = 'b', alpha= 0.2)
ax.plot(np.array(angle_mean[0])-np.array(angle_std[0])+np.pi, np.array(range(len(angle_mean[0])))/25.0, marker = '.', ms = 0.2, linewidth = 0.5, color = 'b', alpha= 0.1)
ax.plot(np.array(angle_mean[0])+np.array(angle_std[0])+np.pi, np.array(range(len(angle_mean[0])))/25.0, marker = '.', ms = 0.2, linewidth = 0.5, color = 'b', alpha= 0.1)
ax.plot(np.array(angle_mean[1])+np.pi/2, np.array(range(len(angle_mean[1])))/25.0, marker = '.', ms = 2, linewidth = 0, color = 'r')
ax.fill_betweenx(np.array(range(len(angle_mean[1])))/25.0,np.array(angle_mean[1])-np.array(angle_std[1])+np.pi/2, np.array(angle_mean[1])+np.array(angle_std[1])+np.pi/2, color = 'r', alpha= 0.2)
ax.plot(np.array(angle_mean[1])-np.array(angle_std[1])+np.pi/2, np.array(range(len(angle_mean[1])))/25.0, marker = '.', ms = 0.2, linewidth = 0.5, color = 'r', alpha= 0.1)
ax.plot(np.array(angle_mean[1])+np.array(angle_std[1])+np.pi/2, np.array(range(len(angle_mean[1])))/25.0, marker = '.', ms = 0.2, linewidth = 0.5, color = 'r', alpha= 0.1)
ax.plot(np.array(angle_mean[2]), np.array(range(len(angle_mean[2])))/25.0, marker = '.', ms = 2, linewidth = 0, color = 'm')
ax.fill_betweenx(np.array(range(len(angle_mean[2])))/25.0,np.array(angle_mean[2])-np.array(angle_std[2]), np.array(angle_mean[2])+np.array(angle_std[2]), color = 'm', alpha= 0.2)
ax.plot(np.array(angle_mean[2])-np.array(angle_std[2]), np.array(range(len(angle_mean[2])))/25.0, marker = '.', ms = 0.2, linewidth = 0.5, color = 'm', alpha= 0.1)
ax.plot(np.array(angle_mean[2])+np.array(angle_std[2]), np.array(range(len(angle_mean[2])))/25.0, marker = '.', ms = 0.2, linewidth = 0.5, color = 'm', alpha= 0.1)
ax.set_xticks(np.pi/180. * np.linspace(0,  360, 36, endpoint=False))
ax.legend(handles=[blue_patch, red_patch, magenta_patch], fontsize=10)
ax.set_ylim([-20,110])
fig.savefig('/home/spinnaker/Documents/projects/Sevilla_2021_TDE_SSL/data/final/closed_loop/'+str(frequency[0])+'Hz/results/polarplot_mean_' +str(frequency[0]) + 'uncropped_Hz.png', dpi=300)


print 'angle_tdev_0'
print np.std(np.degrees(np.array(angle_mean[0][125:])%(2*np.pi)))
print 'angle_tdev_1'
print np.std(np.degrees(np.array(angle_mean[1][125:])%(2*np.pi)))
print 'angle_tdev_2'
print np.std(np.degrees(np.array(angle_mean[2][125:])%(2*np.pi)))
print 'angle_std_0'
print np.mean(np.degrees(np.array(angle_std[0][125:])%(2*np.pi)))
print 'angle_std_1'
print np.mean(np.degrees(np.array(angle_std[1][125:])%(2*np.pi)))
print 'angle_std_2'
print np.mean(np.degrees(np.array(angle_std[2][125:])%(2*np.pi)))



	
