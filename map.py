# -*- coding:utf-8 -*-
import numpy as np
import pyNN.spiNNaker as spynnaker
import csv
import random

from generate_vr_response import generate_vr_response
from generate_spiking_time import generate_spiking_time
from sentence_to_vec import *
from analyse import *

# Testing module
from helper_function import * 
import sys
sys.dont_write_bytecode = True

#
#  SpiNN-3 board configuration
#  @Before: Epochs              -> set to 100ms   
#           Spiking_time_space  -> shape: (50 dimensional 2d-array)
#           
#
#  Global parameters:
params = eval(open("settings.txt").read())
NUM_PN_CELLS = params['NUM_PN_CELLS']
NUM_KC_CELLS = params['NUM_KC_CELLS']

#  Other stuff
TIME_SLOT   = params['TIME_SLOT']
DATA_AMOUNT = params['DATA_AMOUNT']
SHOW0SAVE1  = params['SHOW0SAVE1']

def setupLayer_PN(time_space):
    '''
     PN ─┬─── pn_neuron_01
         ├─── pn_neuron_02
         ├─── pn_neuron_03
         ├─── ...
         └─── pn_neuron_100

     PN was used as input layer
    '''
    input_population = spynnaker.Population(NUM_PN_CELLS,
                                            spynnaker.SpikeSourceArray(spike_times=time_space),
                                            label='PN_population')
    return input_population

def setupLayer_KC():

    '''
                        ┌────── KC_cell_0001
                        ├────── KC_cell_0002       ┌──────> PN_cell_[i]
                KC ─────┼────── KC_cell_0003  <────┼──────> ...
                        ├────── ....               └──────> PN_cell_[k]
                        └────── KC_cell_2000
               1.Each KC neuron map to around ~20 PN_cells
                 which was chosen randomly from 100  of all
               2.By the property of SpiNNaker Board.
                 Each core contains MAX 256 neurons.
                 Hence 2000 KC_neurons will spreads to around ~10 cores
    '''
    kc_population = spynnaker.Population(NUM_KC_CELLS,
                                         spynnaker.IF_curr_exp(),
                                         label='KC_population')
    return kc_population

def setupProjection_PN_KC(pn_population,kc_population):

    connectionList = list()                                        # Connection list between PN and KC
    for each_kc_cell in xrange(NUM_KC_CELLS):

        count         = 6 
        selectedCells = random.sample(xrange(NUM_PN_CELLS),count) 

        for each_pn_cell in selectedCells:
            single_coonection = (each_pn_cell,each_kc_cell)
            connectionList.append(single_coonection)

    pnkcProjection = spynnaker.Projection(pn_population,
                                          kc_population,
                                          spynnaker.FromListConnector(connectionList),
                                          spynnaker.StaticSynapse(weight=5, delay=1.0))
    return pnkcProjection



#
# @Mentioning: Twitter was tranformed into vectors of 100-dimensional 
#              and then transformed into Vr-response, which was 50-dimensional 
#              dimension=50 was set in the "generate_vr.py"
#

def mapping_process():

    ###################################
    SIM_TIME = TIME_SLOT*DATA_AMOUNT ##
    ###################################
    
    twitter_text_vectors = sentence_to_vec()
    orig = get_nearest_neighbor(twitter_text_vectors)
    show_result(orig,1,'./retrived_data/original_classification')
    response_space       = generate_vr_response(twitter_text_vectors)
    spiking_space        = generate_spiking_time(response_space) 
    np.savetxt("./retrived_data/input_time.txt",spiking_space,fmt='%s',delimiter=',',newline='\n')

    spynnaker.setup(timestep=1)
    spynnaker.set_number_of_neurons_per_core(spynnaker.IF_curr_exp, 250)

    pn_population  = setupLayer_PN(spiking_space)
    kc_population  = setupLayer_KC()
    kc_population.record(["spikes"])

    pn_kc_projection  = setupProjection_PN_KC(pn_population,kc_population)
    spynnaker.run(SIM_TIME)

    neo = kc_population.get_data(variables=["spikes"])
    spikeData_original= neo.segments[0].spiketrains
    spynnaker.end()
    return spikeData_original

def analysing_process(spikeData_original):
    
    simplified = get_count(spikeData_original)
    NN_list = get_nearest_neighbor(simplified)
    return NN_list

if(__name__=='__main__'):

    spikeData_original = mapping_process()
    for index in xrange(len(spikeData_original)):
        spikeData_original[index] = np.array(spikeData_original[index])
        spikeData_original[index] = spikeData_original[index].tolist() 
    np.savetxt("./retrived_data/spiking_time.txt",spikeData_original,fmt='%s',delimiter=',',newline='\n')
    NN_list = analysing_process(spikeData_original)
    show_result(NN_list,SHOW0SAVE1)
