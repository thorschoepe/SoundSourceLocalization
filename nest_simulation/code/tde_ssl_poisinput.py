"""
this code was run with python2.7.
It includes the TDEs and Time to Rate network descirbed in the paper Closed-Loop Sound Source Localiztion in Neuromorphic Systems, Schoepe et al. 2023.
You need to include the iaf_psc_lin_semd model in nest before you can run the code.
"""



import nest
import matplotlib.pyplot as pl
import numpy as np
import logging
import random
import math
import os

timefactor = 1.0
simtime = 100000.0
neuron_models = [ "iaf_psc_lin_semd"]
#tau_trig = 2.0
tau_trig = 1.3

distances = [50]
frequencies = np.arange(500.0,1500.0,250.0)
num_directions = 4
num_tdes = 4
num_channels = 1
size_ring = 32
step = 1
cochlea_channels = 64
scale = 0.3
weights_tde = [-90.0, -60.0, -30.0, -10.0]
time_differences = 12.0 # 100 us
res_time_diff = 0.5

# load linear tde model #############################################################
nest.Install("mymodule")


	
def skipper(fname):
    with open(fname) as fin:
        no_comments = (line for line in fin if not line.lstrip().startswith('#'))
        next(no_comments, None) # skip header
        for row in no_comments:
            yield row



for neuron_model in neuron_models: # in case you want to test different neuron models
		for frequency in frequencies: # generate different poisson spike trains with different fredquencies
			print frequency
			for time_difference in np.arange(-time_differences, time_differences, res_time_diff): # generate different time differences
				print frequency
				num_spikestdes = [[] for _ in range(3)]
				num_spikestdes2 = [[] for _ in range(3)]
				spikesdiffs1 = [[] for _ in range(3)]
				spikesdiffs2 = [[] for _ in range(3)]
				for runs in range(3):
					nest.ResetKernel()
					nest.resolution = 0.0001
					
					# generate two poisson spike trains with gaussian distributed inter spike interval #########################################################
					lamb = frequency
					poisson = np.random.poisson(lamb, 100000) 
					ISI = np.random.normal(time_difference,4.0, 100000)
					time_delta = 1000.0/poisson
					events_left = [ [] for i in range(num_channels)]
					events_right = [ [] for i in range(num_channels)]
					time = 20.0
					for i in range(len(time_delta)):
						time += (time_delta[i]*100.0)
						events_left[0].append(round(time, 1))
						events_right[0].append(round(time+ISI[i], 1))
						if time > simtime:
							break
					
					events_left[0] = np.sort(events_left[0])
					events_right[0] = np.sort(events_right[0])
					

							
					print "run " + str(time_differences)
					nest.ResetKernel()

					# Set parameters of neurons ####################################################################################################################
					tde_params = [[] for _ in range(num_tdes)]
					tau_syn_ins = [9.0, 2.0, 0.6, 0.015]
					
					for i in range(num_tdes):
						tde_params[i]= {"C_m": 250.0, "tau_m": 0.01, "t_ref": 0.0, "tau_syn_ex":tau_trig*timefactor, "tau_syn_in": tau_syn_ins[i]*timefactor, # changed taum was 0.01
								"E_L": -70.0, "V_reset": -80.0, "V_m": -70.0, "V_th": -65.0}  
					tde_inhib_params= {"C_m": 250.0, "tau_m": 100.0, "t_ref": 10.0, "tau_syn_ex":tau_trig, "tau_syn_in": 0.3, # changed tref to 10
									"E_L": -70.0, "V_reset": -80.0, "V_m": -70.0, "V_th": -68.0}
					neuron_params= {"C_m": 250.0, "tau_m": 150.0, "t_ref": 10.0, "tau_syn_ex":100.0, "tau_syn_in": 100.0,
									"E_L": -70.0, "V_reset": -70.0, "V_m": -70.0, "V_th": -60.0}
					neuron_params2= {"C_m": 250.0, "tau_m": 150.0, "t_ref": 200.0, "tau_syn_ex":100.0, "tau_syn_in": 100.0,
									"E_L": -70.0, "V_reset": -70.0, "V_m": -70.0, "V_th": -60.0}
					params_input = {'E_L': -65.0, 'C_m': 250.0,'tau_m': 80.0,
									't_ref': 10.0,
									'tau_syn_ex': 60.0,
									'tau_syn_in': 60.0,
									'V_th': -40.0,
									'V_reset': -70.0,
									'V_m': -65.0}
									
					params_ring = {'E_L': -65.0, 
									'C_m': 250.0, # pF
									'tau_m': 60.0,
									't_ref': 10.0,
									'tau_syn_ex': 40.0,
									'tau_syn_in': 100.0,
									'V_th': -40.0,
									'V_reset': -70.0,
									'V_m': -65.0}

					params_inv = {'E_L': -65.0, 
									'C_m': 250.0, # pF
									'tau_m': 100.0,
									't_ref': 10.0,
									'tau_syn_ex': 100.0,
									'tau_syn_in': 100.0,
									'V_th': -50.0,
									'V_reset': -70.0,
									'V_m': -65.0}
									
					params_gi = {'E_L': -65.0, 
									'C_m': 250.0, # pF
									'tau_m': 100.0,
									't_ref': 10.0,
									'tau_syn_ex': 100.0,
									'tau_syn_in': 100.0,
									'V_th': -50.0,
									'V_reset': -70.0,
									'V_m': -65.0}
									
					params_wta = {'E_L': -65.0, 
									'C_m': 250.0, # pF
									'tau_m': 200.0*timefactor,
									't_ref': 200.0*timefactor,
									'tau_syn_ex': 100.0*timefactor,
									'tau_syn_in': 200.0*timefactor,
									'V_th': -50.0,
									'V_reset': -68.0,
									'V_m': -65.0}
									
					nest.SetDefaults(neuron_model,tde_params[0])
					nest.SetDefaults("iaf_psc_exp",neuron_params)

					# create neuron populations #################################################################################################################
					tde1 = [[] for _ in range(num_tdes)]
					tde2 = [[] for _ in range(num_tdes)]
					spikedetectortde1 = [[] for _ in range(num_tdes)]
					spikedetectortde2 = [[] for _ in range(num_tdes)]
					detect_tde1 = [[] for _ in range(num_tdes)]
					detect_tde2 = [[] for _ in range(num_tdes)]

					
						
					for i in range(num_channels):
						for y in range(num_tdes):
							tde1[y].append(nest.Create(neuron_model, 1)) 
							tde2[y].append(nest.Create(neuron_model, 1)) 
							spikedetectortde1[y].append(nest.Create('spike_detector', 1))
							spikedetectortde2[y].append(nest.Create('spike_detector', 1))
							nest.SetStatus(tde1[y][i], tde_params[y])
							nest.SetStatus(tde2[y][i], tde_params[y])
							detect_tde1[y].append(nest.Create('iaf_psc_exp', 1)) 
							detect_tde2[y].append(nest.Create('iaf_psc_exp', 1)) 
							nest.SetStatus(detect_tde1[y][i], neuron_params2)
							nest.SetStatus(detect_tde2[y][i], neuron_params2)



					
					spikedetector_ring = nest.Create('spike_detector', size_ring)
					spikedetector_ring2 = nest.Create('spike_detector', size_ring)
					spikedetector_input = nest.Create('spike_detector', size_ring)
					spikedetector_input2 = nest.Create('spike_detector', size_ring)
					



					# time to rate network ######
					spikedetector_diff1 = nest.Create('spike_detector', 1)
					spikedetector_diff2 = nest.Create('spike_detector', 1)
					# output neurons of time to rate network
					diff1 = nest.Create('iaf_psc_exp', 1) 
					diff2 = nest.Create('iaf_psc_exp', 1)
					nest.SetStatus(diff1, params_wta)
					nest.SetStatus(diff2, params_wta)

						
					# ring attractor######
					input_right = nest.Create('iaf_psc_exp', size_ring)
					input_left = nest.Create('iaf_psc_exp', size_ring)
					ring_right = nest.Create('iaf_psc_exp', size_ring)
					ring_left = nest.Create('iaf_psc_exp', size_ring)
					spike_initial = nest.Create('iaf_psc_exp', 1) # initial spike to set position of ring attractor
					nest.SetStatus(input_right, params_input)
					nest.SetStatus(input_left, params_input)
					nest.SetStatus(ring_right, params_ring)
					nest.SetStatus(ring_left, params_ring)
						
					# create synapses ###############################################################################################################################
					for y in range(num_tdes):
						nest.CopyModel('static_synapse', 'fac'+str(time_difference)+str(y), {'weight' : weights_tde[y], 'delay' : 0.1}) # facailitatory inhibitory was -100
					nest.CopyModel('static_synapse', 'trig'+str(time_difference), {'weight' : 1.0, 'delay' : 0.1}) # trigger excitatory was 500
					nest.CopyModel('static_synapse', 'epg_pen_exc', {'weight' : 1200.0, 'delay' : 0.1}) 
					nest.CopyModel('static_synapse', 'epg_pen_inh', {'weight' : -35.0, 'delay' : 0.1}) 
					nest.CopyModel('static_synapse', 'pen_epg_exc', {'weight' : 500.0, 'delay' : 0.1}) 
					nest.CopyModel('static_synapse', 'invert', {'weight' : 100.0, 'delay' : 0.1}) 
					nest.CopyModel('static_synapse', 'pen_mem_exc', {'weight' : 15.0, 'delay' : 0.1}) 
					nest.CopyModel('static_synapse', 'pen_mem_inh', {'weight' : -15.0, 'delay' : 0.1}) 
					nest.CopyModel('static_synapse', 'tde_lat_inh', {'weight' : -10000.0, 'delay' : 0.1}) 
					nest.CopyModel('static_synapse', 'tde_all_inh', {'weight' : -300.0, 'delay' : 0.1}) 
					nest.CopyModel('static_synapse', 'gi_exc', {'weight' : 20000.0, 'delay' : 0.1}) 
					nest.CopyModel('static_synapse', 'tde_2_dir', {'weight' : 10.0, 'delay' : 0.1}) 
					nest.CopyModel('static_synapse', 'gi_inh', {'weight' : -5000.0, 'delay' : 0.1}) 
					nest.CopyModel('static_synapse', 'pois', {'weight' : 1000.0, 'delay' : 0.1}) 
					nest.CopyModel('static_synapse', 'pois_wta', {'weight' : 4.0, 'delay' : 0.1}) # was 6
					nest.CopyModel('static_synapse', 'inv_inp_inh', {'weight' : -200.0, 'delay' : 0.1}) 
					nest.CopyModel('static_synapse', 'inv_inp_exc', {'weight' : 200.0, 'delay' : 0.1}) 
					nest.CopyModel('static_synapse', 'tde_2_dir_exc', {'weight' : 100.0, 'delay' : 0.1}) 
					nest.CopyModel('static_synapse', 'tde_2_dir_inh', {'weight' : -100.0, 'delay' : 0.1}) 
					nest.CopyModel('static_synapse', 'wta_exc', {'weight' : 200.0, 'delay' : 0.1}) 
					nest.CopyModel('static_synapse', 'wta_exc2', {'weight' : 180.0, 'delay' : 0.1}) 
					nest.CopyModel('static_synapse', 'wta_inh', {'weight' : -40.0, 'delay' : 0.1}) 
					nest.CopyModel('static_synapse', 'wta_inh2', {'weight' : -50.0, 'delay' : 0.1}) 
					nest.CopyModel('static_synapse', 'wta_lat', {'weight' : -1000.0, 'delay' : 0.1}) 
					nest.CopyModel('static_synapse', 'wta_diff3', {'weight' : -3000.0, 'delay' : 0.1}) 
					nest.CopyModel('static_synapse', 'tde_inh', {'weight' : -3000.0, 'delay' : 0.1}) 



					# create spike generator ###############################################################################################################################
					
					spikegenerator = []
					spikegenerator2 = [[] for _ in range(num_tdes)]
					spikegenerator3 = [[] for _ in range(num_tdes)]
					spikegenerator4 = []
					spikegenerator5 = []
					spikegenerator6 = []
					
					spikegenerator7 = []
					spikegenerator8 = []
					
					spikegenerator_tde_inhib1 = []
					spikegenerator_tde_inhib2 = []
					spikegenerator_tde_inhib3 = []
					spikegenerator_tde_inhib4 = []
					
					pois_tde1 = [[] for _ in range(num_tdes)]
					pois_tde2 = [[] for _ in range(num_tdes)]
					pois_tde3 = [[] for _ in range(num_tdes)]
					pois_tde4 = [[] for _ in range(num_tdes)]
					

					for i in range(num_channels):
						spikegenerator.append(nest.Create('spike_generator')) # create a spike generator
						spikegenerator7.append(nest.Create('spike_generator'))
						spikegenerator8.append(nest.Create('spike_generator'))
						spikegenerator_tde_inhib1.append(nest.Create('spike_generator'))
						spikegenerator_tde_inhib2.append(nest.Create('spike_generator'))
						nest.SetStatus(spikegenerator_tde_inhib1[i], {'spike_times': events_right[i]})
						nest.SetStatus(spikegenerator_tde_inhib2[i], {'spike_times': events_left[i]})
						spikegenerator_tde_inhib3.append(nest.Create('spike_generator'))
						spikegenerator_tde_inhib4.append(nest.Create('spike_generator'))
						nest.SetStatus(spikegenerator_tde_inhib3[i], {'spike_times': events_right[i]})
						nest.SetStatus(spikegenerator_tde_inhib4[i], {'spike_times': events_left[i]})
						for y in range(num_tdes):
							spikegenerator2[y].append(nest.Create('spike_generator'))
							spikegenerator3[y].append(nest.Create('spike_generator'))
						spikegenerator4.append(nest.Create('spike_generator'))
						spikegenerator5.append(nest.Create('spike_generator'))
						spikegenerator6.append(nest.Create('spike_generator'))
						nest.SetStatus(spikegenerator[i], {'spike_times': events_left[i]}) # let it spike# 
						for y in range(num_tdes):							
							nest.SetStatus(spikegenerator2[y][i], {'spike_times': events_right[i]}) # let it spike# 
							nest.SetStatus(spikegenerator3[y][i], {'spike_times': events_left[i]}) # let it spike# 
						nest.SetStatus(spikegenerator4[i], {'spike_times': events_right[i]}) # let it spike# 
						nest.SetStatus(spikegenerator5[i], {'spike_times': events_left[i]}) # let it spike# 
						nest.SetStatus(spikegenerator6[i], {'spike_times': events_right[i]}) # let it spike# 
						nest.SetStatus(spikegenerator7[i], {'spike_times': events_right[i]})
						nest.SetStatus(spikegenerator8[i], {'spike_times': events_left[i]})
						
					spikegenerator_initial = nest.Create('spike_generator')
					nest.SetStatus(spikegenerator_initial, {'spike_times': np.arange(1.0, 20.0, 1.0)}) # let it spike# 
					pois_ring = nest.Create('poisson_generator', params={'rate': 100.0, 'start': 0.0,'stop': 100000.0}) # keeps up activity of ring attractor
					
					# add random spikes to input
					frequ_random = 500
					amount_random = frequ_random/(100000.0/simtime)
					random_events_right = np.round_(np.sort(np.random.random(amount_random)*simtime),1)
					random_events_left = np.round_(np.sort(np.random.random(amount_random)*simtime),1)
				
					
					for y in range(num_tdes):
						pois_tde1[y] = (nest.Create('spike_generator'))
						pois_tde2[y] = (nest.Create('spike_generator'))
						pois_tde3[y] = (nest.Create('spike_generator'))
						pois_tde4[y] = (nest.Create('spike_generator'))
						nest.SetStatus(pois_tde1[y], {'spike_times': random_events_right})
						nest.SetStatus(pois_tde2[y], {'spike_times': random_events_left})
						nest.SetStatus(pois_tde3[y], {'spike_times': random_events_right})
						nest.SetStatus(pois_tde4[y], {'spike_times': random_events_left})


					# connections ###########################################################################################################################################
					for i in range(num_channels):
						for y in range(num_tdes):
							nest.Connect(spikegenerator[i], tde1[y][i],'one_to_one',{'model' : 'trig'+str(time_difference),'weight': 500.0,  'delay':0.1}) 
							nest.Connect(spikegenerator2[y][i], tde1[y][i],'one_to_one',{'model' : 'fac'+str(time_difference)+str(y),'weight': weights_tde[y],  'delay':0.1}) 
							nest.Connect(spikegenerator3[y][i], tde2[y][i],'one_to_one',{'model' : 'fac'+str(time_difference)+str(y),'weight': weights_tde[y],  'delay':0.1}) 
							nest.Connect(spikegenerator4[i], tde2[y][i],'one_to_one',{'model' : 'trig'+str(time_difference),'weight': 500.0,  'delay':0.1})
							nest.Connect(tde1[y][i], detect_tde1[y][i],'one_to_one',{'model' : 'pois','weight': 200.0,  'delay':0.1}) 
							nest.Connect(tde2[y][i], detect_tde2[y][i],'one_to_one',{'model' : 'pois','weight': 200.0,  'delay':0.1}) 
							nest.Connect(detect_tde1[y][i], spikedetectortde1[y][i])
							nest.Connect(detect_tde2[y][i], spikedetectortde2[y][i])
							nest.Connect(pois_tde1[y], tde1[y][i],'one_to_one',{'model' : 'trig'+str(time_difference),'weight': 500.0,  'delay':0.1})
							nest.Connect(pois_tde2[y], tde2[y][i],'one_to_one',{'model' : 'trig'+str(time_difference),'weight': 500.0,  'delay':0.1})	
							nest.Connect(pois_tde3[y], tde2[y][i],'one_to_one',{'model' : 'fac'+str(time_difference)+str(y),'weight': weights_tde[y],  'delay':0.1})
							nest.Connect(pois_tde4[y], tde1[y][i],'one_to_one',{'model' : 'fac'+str(time_difference)+str(y),'weight': weights_tde[y],  'delay':0.1})						
							

						

						
					nest.Connect(pois_ring, input_left,'all_to_all',{'model' : 'pois','weight': 100.0,  'delay':0.1})
					nest.Connect(pois_ring, input_right,'all_to_all',{'model' : 'pois','weight': 100.0,  'delay':0.1})
					
					
					weight_exc = 500.0/num_channels
					weight_inh = -500.0/num_channels
					
					pois_background =  nest.Create('poisson_generator', params={'rate': 0.0})
					pois_background2 =  nest.Create('poisson_generator', params={'rate': 0.0})
					
					for i in range(num_channels):
						nest.Connect(detect_tde1[0][i], diff1,'one_to_one',{'model' : 'wta_inh','weight': -200.0,  'delay':0.1}) # -100
						nest.Connect(detect_tde1[1][i], diff1,'one_to_one',{'model' : 'wta_inh','weight': -50.0,  'delay':0.1}) # -100
						nest.Connect(detect_tde1[2][i], diff1,'one_to_one',{'model' : 'wta_inh','weight': -50.0,  'delay':0.1}) #-200
						nest.Connect(detect_tde1[3][i], diff1,'one_to_one',{'model' : 'wta_exc','weight': 200.0,  'delay':0.1}) # 200
						nest.Connect(detect_tde2[3][i], diff1,'one_to_one',{'model' : 'wta_exc','weight': 200.0,  'delay':0.1}) #200
						
						
						nest.Connect(detect_tde2[0][i], diff2,'one_to_one',{'model' : 'wta_inh','weight': -200.0,  'delay':0.1})
						nest.Connect(detect_tde2[1][i], diff2,'one_to_one',{'model' : 'wta_inh','weight': -50.0,  'delay':0.1})
						nest.Connect(detect_tde2[2][i], diff2,'one_to_one',{'model' : 'wta_inh','weight': -50.0,  'delay':0.1})
						nest.Connect(detect_tde2[3][i], diff2,'one_to_one',{'model' : 'wta_exc','weight': 200.0,  'delay':0.1})
						nest.Connect(detect_tde1[3][i], diff2,'one_to_one',{'model' : 'wta_exc','weight': 200.0,  'delay':0.1})
					
					nest.Connect(diff1, diff2,'all_to_all',{'model' : 'inv_inp_exc','weight': -200.0,  'delay':0.1})
					nest.Connect(diff2, diff1,'all_to_all',{'model' : 'inv_inp_exc','weight': -200.0,  'delay':0.1})

						
					
					nest.Connect(diff1, spikedetector_diff1)
					nest.Connect(diff2, spikedetector_diff2)
					

				
						
					### central complex ######
					nest.Connect(spikegenerator_initial[0], ring_right[int(size_ring/4.0)],'pen_epg_exc')
					#e_pg to p_en
					nest.Connect(ring_right, input_right,'one_to_one',{'model' : 'epg_pen_exc','weight': 1000.0,  'delay':0.1}) 
					nest.Connect(ring_left, input_left,'one_to_one',{'model' : 'epg_pen_exc','weight': 1000.0,  'delay':0.1})
					# epg pen inhib
					nest.Connect(ring_right, input_right,'all_to_all',{'model' : 'epg_pen_inh','weight': -1500.0/(size_ring/2.0),  'delay':0.1}) 
					nest.Connect(ring_left, input_left,'all_to_all',{'model' : 'epg_pen_inh','weight': -1500.0/(size_ring/2.0),  'delay':0.1})
					nest.Connect(ring_right, input_left,'all_to_all',{'model' : 'epg_pen_inh','weight': -1500.0/(size_ring/2.0),  'delay':0.1})
					nest.Connect(ring_left, input_right,'all_to_all',{'model' : 'epg_pen_inh','weight': -1500.0/(size_ring/2.0),  'delay':0.1})
					# pen epg exc
					for i in range(size_ring):
						nest.Connect(input_right[(i+1)%size_ring], ring_right[i],'pen_epg_exc')
						nest.Connect(input_right[(i+1)%size_ring], ring_left[i],'pen_epg_exc')
						nest.Connect(input_left[i], ring_right[(i+1)%size_ring],'pen_epg_exc')
						nest.Connect(input_left[i], ring_left[(i+1)%size_ring],'pen_epg_exc')
						
					nest.Connect(ring_right, spikedetector_ring,'one_to_one')
					nest.Connect(ring_left, spikedetector_ring2,'one_to_one')
					nest.Connect(input_right, spikedetector_input, 'one_to_one')
					nest.Connect(input_left, spikedetector_input2, 'one_to_one')
					
					# run sim ##############################################################################################################################################
					print "start sim"
					print frequency
					print time_difference
					
					nest.Simulate(float(simtime)) # run the simulation# 

					#read out recording time and voltage from voltmeter and plot them ######################################################################################
					spikestde = [[] for i in range(num_tdes)]
					spikestde2 = [[] for i in range(num_tdes)]
					num_spikestde = [[] for i in range(num_tdes)]
					num_spikestde2 = [[] for i in range(num_tdes)]
					number_spikes_1 = 0
					number_spikes_2 = 0
					for i in range(num_channels):
						for y in range(num_tdes):
							spikestde[y].append(nest.GetStatus(spikedetectortde1[y][i])[0]['events']['times'])
							spikestde2[y].append(nest.GetStatus(spikedetectortde2[y][i])[0]['events']['times'])
							num_spikestde[y].append(len(nest.GetStatus(spikedetectortde1[y][i])[0]['events']['times']))
							num_spikestde2[y].append(len(nest.GetStatus(spikedetectortde2[y][i])[0]['events']['times']))

			
					spikesdiff1 = nest.GetStatus(spikedetector_diff1)[0]['events']['times']
					spikesdiff2 = nest.GetStatus(spikedetector_diff2)[0]['events']['times']

					
					spikesring = [[] for _ in range(size_ring)]
					spikesring2 = [[] for _ in range(size_ring)]
					spikesinput = [[] for _ in range(size_ring)]
					spikesinput2 = [[] for _ in range(size_ring)]
					
					for i in range(size_ring):
						spikesring[i] = nest.GetStatus(spikedetector_ring)[i]['events']['times']
						spikesring2[i] = nest.GetStatus(spikedetector_ring2)[i]['events']['times']
						spikesinput[i] = nest.GetStatus(spikedetector_input)[i]['events']['times']
						spikesinput2[i] = nest.GetStatus(spikedetector_input2)[i]['events']['times']
						
						

					frequency = int(frequency)
					
					spikesdiffs1[runs] = spikesdiff1
					spikesdiffs2[runs] = spikesdiff2
					num_spikestdes[runs] = num_spikestde
					num_spikestdes2[runs] = num_spikestde2

					# save events ##################################################################################################
					directory = "../"
					np.save(directory+str(frequency)+'Hz_itde2/spikes_tde_npy/spikesring_fr_'+str(frequency)+neuron_model+'tau_trig'+str(tau_trig)+'pos_'+str(time_difference)+'.npy', spikesring)
					np.save(directory+str(frequency)+'Hz_itde2/spikes_tde_npy/spikesring2_fr_'+str(frequency)+neuron_model+'tau_trig'+str(tau_trig)+'pos_'+str(time_difference)+'.npy', spikesring2)
					np.save(directory+str(frequency)+'Hz_itde2/spikes_tde_npy/spikesinput_fr_'+str(frequency)+neuron_model+'tau_trig'+str(tau_trig)+'pos_'+str(time_difference)+'.npy', spikesinput)
					np.save(directory+str(frequency)+'Hz_itde2/spikes_tde_npy/spikesinput2_fr_'+str(frequency)+neuron_model+'tau_trig'+str(tau_trig)+'pos_'+str(time_difference)+'.npy', spikesinput2)
					np.save(directory+str(frequency)+'Hz_itde2/spikes_tde_npy/spikestde_fr_'+str(frequency)+neuron_model+'tau_trig'+str(tau_trig)+'pos_'+str(time_difference)+'.npy', spikestde)
					np.save(directory+str(frequency)+'Hz_itde2/spikes_tde_npy/spikestde2_fr_'+str(frequency)+neuron_model+'tau_trig'+str(tau_trig)+'pos_'+str(time_difference)+'.npy', spikestde2)

					np.save(directory+str(frequency)+'Hz_itde2/spikes_tde_npy/num_spikestde_fr_'+str(frequency)+neuron_model+'tau_trig'+str(tau_trig)+'pos_'+str(time_difference)+'.npy', num_spikestdes)
					np.save(directory+str(frequency)+'Hz_itde2/spikes_tde_npy/num_spikestde2_fr_'+str(frequency)+neuron_model+'tau_trig'+str(tau_trig)+'pos_'+str(time_difference)+'.npy', num_spikestdes2)

				
					np.save(directory+str(frequency)+'Hz_itde2/spikes_tde_npy/spikesdiff1_fr_'+str(frequency)+neuron_model+'tau_trig'+str(tau_trig)+'pos_'+str(time_difference)+'.npy', spikesdiffs1)
					np.save(directory+str(frequency)+'Hz_itde2/spikes_tde_npy/spikesdiff2_fr_'+str(frequency)+neuron_model+'tau_trig'+str(tau_trig)+'pos_'+str(time_difference)+'.npy', spikesdiffs2)


				







    




