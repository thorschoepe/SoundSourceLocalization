
# NAS TDE configuration: channel1: first eight TDEs, right: first four TDEs, left: next four TDES, channel2: second eight TDEs, and so on


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import spynnaker8 as p 
from matplotlib import gridspec
from quantities import ms


def rasterplot(spiketimes,populations_size, simtime, x_labels, y_labels, path_save):
	
	number_subplots = len(spiketimes)
	ax = [[] for _ in range(number_subplots+1)]
	
	
	fig = plt.figure(figsize =(6,4), dpi=300)
	plt.rc("font", size=5)
	

	
		
	
	for num in range(1, number_subplots+1):
		ax[num] = fig.add_subplot(int((number_subplots+1)/2),2, num)
		ax[num].set_ylabel(y_labels[num-1])
		ax[num].set_xlabel(x_labels)
		ax[num].set_ylim([-1,populations_size[num-1]])
		ax[num].set_xlim([0,simtime])
		for i in range(len(spiketimes[num-1].segments[0].spiketrains)):
			ax[num].scatter(spiketimes[num-1].segments[0].spiketrains[i], [i]* len(spiketimes[num-1].segments[0].spiketrains[i]), s=0.01, c = 'b')
			
	fig.tight_layout()
		
	fig.savefig(path_save)
	
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
		#ax.set_xlim(x_lim)
		ax.hist(np.array(spikes).T.flatten(), simtime/1000, density=False, alpha = 0.5)
			
	fig.tight_layout()
		
	fig.savefig(path_save)
	plt.close('all')





#print spikes_left



simtime = 40000 # Simulation time [ms]
timestep = 1
p.setup(timestep) # Simulate with 1 ms time step



#p.set_number_of_neurons_per_core(p.IF_curr_exp, 60)

class Size:
	pass
	
	fpga = 512
	e_pg = p_en = 64 
	edge_dvs = 128
	pop_dvs = edge_dvs**2
	fpga = 512


# cell parameters ##################################################################################################################################

class CellParams:
	pass
	
	p_en =  {'cm'  : 0.25,                               
	   'i_offset'  : 0.0, # was 3.0                             
	   'tau_m'     : 80.0,                             
	   'tau_syn_E' : 60,    
	   'tau_syn_I' : 100, # was 100  
	   'v_reset'   : -70.0,                            
	   'v_rest'    : -65.0,                            
	   'v_thresh'  : -40.0,  
	   'tau_refrac' :  1 
	   }
	
	e_pg =  {'cm'        : 0.25,                               
	   'i_offset'  : 0.0,                            
	   'tau_m'     : 60.0,                            
	   'tau_syn_E' : 40,    
	   'tau_syn_I' : 100,    
	   'v_reset'   : -70.0,                            
	   'v_rest'    : -65.0,                            
	   'v_thresh'  : -40.0,  
	   'tau_refrac' :  1 
	   }
	lif = {'cm': 0.25,
		'i_offset': 0.0,
		'tau_m': 20.0,
		'tau_refrac': 2.0,
		'tau_syn_E': 20.0,
		'tau_syn_I': 20.0,
		'v_reset': -68.0,
		'v_rest': -65.0,
		'v_thresh': -50.0
		}
	lif2 = {'cm': 0.25,
		'i_offset': 0.0,
		'tau_m': 20.0,
		'tau_refrac': 1.0,
		'tau_syn_E': 10.0,
		'tau_syn_I': 10.0,
		'v_reset': -68.0,
		'v_rest': -65.0,
		'v_thresh': -50.0
		}
	wta = {'cm': 0.25,
		'i_offset': 0.0,
		'tau_m': 100.0,
		'tau_refrac': 1.0,
		'tau_syn_E': 20.0,
		'tau_syn_I': 200.0,
		'v_reset': -68.0,
		'v_rest': -65.0,
		'v_thresh': -50.0
		}
	gi = {'cm': 0.25,
		'i_offset': 0.0,
		'tau_m': 20.0,
		'tau_refrac': 1.0,
		'tau_syn_E': 20.0, # was 100
		'tau_syn_I': 20.0,
		'v_reset': -68.0,
		'v_rest': -65.0,
		'v_thresh': -50.0
		}
	out = {'cm': 0.25,
		'i_offset': 0.0,
		'tau_m': 20.0,
		'tau_refrac': 1.0,
		'tau_syn_E': 10.0,
		'tau_syn_I': 20.0, 
		'v_reset': -68.0,
		'v_rest': -65.0,
		'v_thresh': -50.0
		}
	wta2 = {'cm': 0.25,
		'i_offset': 0.0,
		'tau_m': 200.0,
		'tau_refrac': 1.0,
		'tau_syn_E': 10.0,
		'tau_syn_I': 10.0,
		'v_reset': -68.0,
		'v_rest': -65.0,
		'v_thresh': -50.0
		}
	onset = {'cm': 0.25,
		'i_offset': 0.0,
		'tau_m': 200.0,
		'tau_refrac': 1.0,
		'tau_syn_E': 10.0,
		'tau_syn_I': 100.0,
		'v_reset': -68.0,
		'v_rest': -65.0,
		'v_thresh': -50.0
		}
	onset_inhib = {'cm': 0.25,
		'i_offset': 0.0,
		'tau_m': 200.0,
		'tau_refrac': 1.0,
		'tau_syn_E': 10.0,
		'tau_syn_I': 10.0,
		'v_reset': -68.0,
		'v_rest': -65.0,
		'v_thresh': -40.0
		}
	tde_in = {'cm': 0.25,
		'i_offset': 0.0,
		'tau_m': 100.0,
		'tau_refrac': 2.0,
		'tau_syn_E': 20.0,
		'tau_syn_I': 20.0,
		'v_reset': -68.0,
		'v_rest': -65.0,
		'v_thresh': -50.0
		}



# connections ################################################################################################################################

bump_to_output_ratio = 4 # was 9
num_channels = divisor =  1
num_tdes = 4   
class Connect:
	p_en_l_2_e_pg = [(i, (i+1), 0.5, 1) for i in range(Size.p_en-1)]
	p_en_r_2_e_pg = [((i+1),i, 0.5, 1) for i in range(Size.p_en-1)]
	bump_2_output = [[i,260+ i*bump_to_output_ratio,1,1] for i in range(0,Size.e_pg,1)]
	bump_2_output_2 = [[i,0+ i*bump_to_output_ratio,1,1] for i in range(0,Size.e_pg,1)]
	tde_2_diff_l_inh = [[i,i+1,1.0,1] for i in range(3)]
	tde_2_diff_l_exc = [[i,i,1.0,1] for i in range(1,4,1)]
	tde_2_diff_r_inh = [[i,i+1-4,1.0,1] for i in range(4,7,1)]
	tde_2_diff_r_exc = [[i,i-4,1.0,1] for i in range(5,8,1)]
	tde_2_out_r = []
	tde_2_out_l = []
	for z in range(num_channels):
		for i in range(1,4,1):
			tde_2_out_r.append([i+(z*num_tdes*2),0,50/divisor,1])
			tde_2_out_l.append([i+4+(z*num_tdes*2),0,50/divisor,1])
			
	print tde_2_out_r
	tde_2_wta_r = [[i,0,10,1] for i in range(0,4,1)]
	tde_2_wta_l = [[i+4,0,10,1] for i in range(0,4,1)]
	middle_filter = []
	middle_lateral = []
	for i in range(Size.e_pg):
		middle_lateral.append([i,i+1,2,1])
		middle_lateral.append([i+1,i,2,1])
		middle_lateral.append([i,i+2,2,1])
		middle_lateral.append([i+2,i,2,1])
		middle_lateral.append([i,i+3,2,1])
		middle_lateral.append([i+3,i,2,1])
		middle_lateral.append([i,i+4,2,1])
		middle_lateral.append([i+4,i,2,1])
						
	for filtr in range(1,5,1):
		for i in range(Size.p_en):
			middle_filter.append([i, i+filtr, 0.5/filtr,1])
			middle_filter.append([i+filtr, i, 0.5/filtr,1])
	

	
print Connect.tde_2_diff_r_inh
print Connect.tde_2_diff_r_exc	
print Connect.tde_2_diff_l_inh
print Connect.tde_2_diff_l_exc	
	



# populations #################################################################################################################################
def get_updated_params(params):
	params.update({"spinnaker_link_id": 0})
	return params
	
class Populations:
	# ring attractor
	spike_source = p.Population(1, p.SpikeSourceArray, {'spike_times': [[5, 10, 15]]}, label = "input_pop")
	poisson_ring = p.Population(Size.p_en, p.SpikeSourcePoisson, {'rate':10}, label = "poisson")
	p_en_right = p.Population(Size.p_en,p.IF_curr_exp,CellParams.p_en,label='p-en_right')
	p_en_left = p.Population(Size.p_en,p.IF_curr_exp,CellParams.p_en,label='p-en_left')
	e_pg_right = p.Population(Size.fpga,p.IF_curr_exp,CellParams.e_pg,label='e-pg')
	e_pg_left = p.Population(Size.fpga,p.IF_curr_exp,CellParams.e_pg,label='e-pg')
	
	wta = p.Population(Size.p_en,p.IF_curr_exp,CellParams.wta,label='wta')
	
	# tde sound input
	tde_out = p.Population(size=64, cellclass=p.external_devices.ExternalCochleaDevice(spinnaker_link_id=1, board_address=None, cochlea_key=0x200, cochlea_n_channels=p.external_devices.ExternalCochleaDevice.CHANNELS_64, cochlea_type=p.external_devices.ExternalCochleaDevice.TYPE_MONO, cochlea_polarity=p.external_devices.ExternalCochleaDevice.POLARITY_UP), label="ExternalCochlea")
	retina = p.Population(Size.pop_dvs, p.external_devices.ExternalFPGARetinaDevice, get_updated_params({'retina_key': 0xfefe,'mode': p.external_devices.ExternalFPGARetinaDevice.MODE_128,'polarity': p.external_devices.ExternalFPGARetinaDevice.UP_POLARITY}),label='External_retina')
	tde_in = p.Population(64,p.IF_curr_exp,CellParams.lif,label='tde_in')
	out_r = p.Population(1,p.IF_curr_exp,CellParams.out,label='out_r')
	out_l = p.Population(1,p.IF_curr_exp,CellParams.out,label='out_l')
	wta_r = p.Population(1,p.IF_curr_exp,CellParams.wta2,label='wta_r')
	wta_l = p.Population(1,p.IF_curr_exp,CellParams.wta2,label='wta_l')
	
	# motor control
	output = p.Population(Size.fpga, p.IF_curr_exp, CellParams.lif, label='output_layer')
	gi = p.Population(1, p.IF_curr_exp, CellParams.gi, label='gi')
	
	onset = p.Population(Size.p_en,p.IF_curr_exp,CellParams.onset,label='wta')
	onset_inhib = p.Population(Size.p_en,p.IF_curr_exp,CellParams.onset_inhib,label='wta')
	
	poisson_out = p.Population(1, p.SpikeSourcePoisson, {'rate':700}, label = "poisson_out")


	 

# projections #########################################################################################################################################

# intializing bump position
p.Projection(Populations.spike_source, Populations.p_en_right, p.FromListConnector([[0, Size.p_en/2, 10, 1]]), p.StaticSynapse(), receptor_type='excitatory')

p.Projection(Populations.tde_out, Populations.tde_in, p.OneToOneConnector(),p.StaticSynapse(weight=0.3, delay = 1), receptor_type = 'excitatory')



for i in range(num_channels):
	p.Projection(Populations.tde_in, Populations.out_r, p.FromListConnector([[3+((num_channels-1)*num_tdes*2),0,200/divisor,1]]),p.StaticSynapse(), receptor_type='excitatory') 
	p.Projection(Populations.tde_in, Populations.out_l, p.FromListConnector([[3+((num_channels-1)*num_tdes*2),0,200/divisor,1]]),p.StaticSynapse(), receptor_type='excitatory') 
	p.Projection(Populations.tde_in, Populations.out_r, p.FromListConnector([[7+((num_channels-1)*num_tdes*2),0,200/divisor,1]]),p.StaticSynapse(), receptor_type='excitatory') 
	p.Projection(Populations.tde_in, Populations.out_l, p.FromListConnector([[7+((num_channels-1)*num_tdes*2),0,200/divisor,1]]),p.StaticSynapse(), receptor_type='excitatory') 

	p.Projection(Populations.tde_in, Populations.out_r, p.FromListConnector([[0+((num_channels-1)*num_tdes*2),0,200/divisor,1]]), p.StaticSynapse(), receptor_type='inhibitory') # was 150
	p.Projection(Populations.tde_in, Populations.out_l, p.FromListConnector([[4+((num_channels-1)*num_tdes*2),0,200/divisor,1]]), p.StaticSynapse(), receptor_type='inhibitory') # was 150
	p.Projection(Populations.tde_in, Populations.out_r, p.FromListConnector(Connect.tde_2_out_l), p.StaticSynapse(), receptor_type='inhibitory')
	p.Projection(Populations.tde_in, Populations.out_l, p.FromListConnector(Connect.tde_2_out_r), p.StaticSynapse(), receptor_type='inhibitory')





#p.Projection(Populations.wta_r, Populations.out_r, p.OneToOneConnector(),p.StaticSynapse(weight=100.0, delay = 1), receptor_type='inhibitory')
#p.Projection(Populations.wta_l, Populations.out_l, p.OneToOneConnector(),p.StaticSynapse(weight=100.0, delay = 1), receptor_type='inhibitory')

p.Projection(Populations.out_r, Populations.out_l, p.AllToAllConnector(),p.StaticSynapse(weight=80.0, delay = 1), receptor_type='inhibitory') # was 10
p.Projection(Populations.out_l, Populations.out_r, p.AllToAllConnector(),p.StaticSynapse(weight=80.0, delay = 1), receptor_type='inhibitory') # was 10
		
		
# out to ring attractor
p.Projection(Populations.out_l, Populations.p_en_right, p.AllToAllConnector(),p.StaticSynapse(weight = 0.8, delay = 1), receptor_type='excitatory') # was 1.0
p.Projection(Populations.out_r, Populations.p_en_left, p.AllToAllConnector(),p.StaticSynapse(weight = 0.8, delay = 1), receptor_type='excitatory')



## ring attractor
p.Projection(Populations.poisson_ring, Populations.p_en_right, p.OneToOneConnector(),p.StaticSynapse(weight = 1.0, delay = 1), receptor_type='excitatory')
p.Projection(Populations.poisson_ring, Populations.p_en_left, p.OneToOneConnector(),p.StaticSynapse(weight = 1.0, delay = 1), receptor_type='excitatory')

p.Projection(Populations.e_pg_right, Populations.p_en_right, p.OneToOneConnector(),p.StaticSynapse(weight = 1.2, delay = 1), receptor_type='excitatory')
p.Projection(Populations.e_pg_left, Populations.p_en_left, p.OneToOneConnector(),p.StaticSynapse(weight = 1.2, delay = 1), receptor_type='excitatory')
##projections from e-pg to gi and from gi to p-en
p.Projection(Populations.e_pg_right, Populations.p_en_right, p.AllToAllConnector(),p.StaticSynapse(weight = 1.3/(float(Size.e_pg)/2.0), delay = 1), receptor_type='inhibitory') # was 0.2
p.Projection(Populations.e_pg_right, Populations.p_en_left, p.AllToAllConnector(),p.StaticSynapse(weight = 1.3/(float(Size.e_pg)/2.0), delay = 1), receptor_type='inhibitory')
p.Projection(Populations.e_pg_left, Populations.p_en_right, p.AllToAllConnector(),p.StaticSynapse(weight = 1.3/(float(Size.e_pg)/2.0), delay = 1), receptor_type='inhibitory')
p.Projection(Populations.e_pg_left, Populations.p_en_left, p.AllToAllConnector(),p.StaticSynapse(weight = 1.3/(float(Size.e_pg)/2.0), delay = 1), receptor_type='inhibitory')
## p-en to e-pg
p.Projection(Populations.p_en_right, Populations.e_pg_right, p.FromListConnector(Connect.p_en_r_2_e_pg), p.StaticSynapse(), receptor_type='excitatory')
p.Projection(Populations.p_en_right, Populations.e_pg_left, p.FromListConnector(Connect.p_en_r_2_e_pg), p.StaticSynapse(), receptor_type='excitatory')
p.Projection(Populations.p_en_left, Populations.e_pg_right, p.FromListConnector(Connect.p_en_l_2_e_pg), p.StaticSynapse(), receptor_type='excitatory')
p.Projection(Populations.p_en_left, Populations.e_pg_left, p.FromListConnector(Connect.p_en_l_2_e_pg), p.StaticSynapse(), receptor_type='excitatory')




p.Projection(Populations.onset, Populations.output, p.FromListConnector(Connect.bump_2_output), p.StaticSynapse(), receptor_type='excitatory')
p.Projection(Populations.onset, Populations.output, p.FromListConnector(Connect.bump_2_output_2), p.StaticSynapse(), receptor_type='excitatory')



p.Projection(Populations.p_en_right, Populations.onset,p.FromListConnector(Connect.middle_filter), p.StaticSynapse(), receptor_type='excitatory')
p.Projection(Populations.p_en_left, Populations.onset,p.FromListConnector(Connect.middle_filter), p.StaticSynapse(), receptor_type='excitatory')
p.Projection(Populations.onset, Populations.onset,p.FromListConnector(Connect.middle_lateral), p.StaticSynapse(), receptor_type='inhibitory')

p.Projection(Populations.onset, Populations.gi, p.AllToAllConnector(),p.StaticSynapse(weight = 4.0, delay = 1), receptor_type='excitatory')	# was 3.0
p.Projection(Populations.gi, Populations.onset, p.AllToAllConnector(),p.StaticSynapse(weight = 4.0, delay = 1), receptor_type='inhibitory') # was 3.0










Populations.p_en_right.record(['spikes'])
Populations.p_en_left.record(['spikes'])
Populations.e_pg_right.record(['spikes'])
Populations.e_pg_left.record(['spikes'])
Populations.tde_in.record(['spikes'])
Populations.out_r.record(['spikes'])
Populations.out_l.record(['spikes'])
Populations.wta.record(['spikes'])
Populations.onset.record(['spikes'])

p.external_devices.activate_live_output_to(Populations.output, Populations.retina)

## run Simulation
###############################################################################################################################################################
p.run(simtime) 

	
class Spikes:
	pass	
	p_en_right=Populations.p_en_right.get_data(['spikes'])
	p_en_left=Populations.p_en_left.get_data(['spikes']) 
	e_pg_right=Populations.e_pg_right.get_data(['spikes']) 
	e_pg_left=Populations.e_pg_left.get_data(['spikes']) 
	tde_in=Populations.tde_in.get_data(['spikes']) 
	angle_difference_r=Populations.out_r.get_data(['spikes']) 
	angle_difference_l=Populations.out_l.get_data(['spikes']) 
	wta=Populations.wta.get_data(['spikes']) 
	onset=Populations.onset.get_data(['spikes']) 
	tde_in_right = []
	tde_in_left = []

Spikes.tde_in_right = Spikes.tde_in.segments[0].spiketrains[0:4]
Spikes.tde_in_left = Spikes.tde_in.segments[0].spiketrains[4:8]

	
### plots ########################################################################################################################################################

spikedata = [Spikes.p_en_right, Spikes.p_en_left, Spikes.e_pg_right,Spikes.e_pg_left,Spikes.tde_in, Spikes.angle_difference_r, Spikes.angle_difference_l, Spikes.onset]
size_populations = [Size.p_en, Size.p_en, Size.e_pg, Size.e_pg, 64, 1, 1, Size.p_en]
y_labels = ['p-en r', 'p-en l','e-pg r', 'e-pg l', 'tde in', 'diff r', 'diff l', 'onset']
path = '/home/spinnaker/Documents/projects/Sevilla_2021_TDE_SSL/data/closed_loop_SSL.png'
rasterplot(spikedata,size_populations, simtime,'time (ms)', y_labels, path)

size_populations = [1,1]
x_lim = [0,120000]
simtime = 120000
y_labels = ['tde', 'tde']
spikedata_hist = [Spikes.tde_in_right, Spikes.tde_in_left]
path = '/home/spinnaker/Documents/projects/Sevilla_2021_TDE_SSL/data/closed_loop_SSL_hist.png'
histogram_overlay(spikedata_hist,size_populations, simtime,'time (ms)', y_labels, path, x_lim)


p.end()






