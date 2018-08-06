# -*- coding:utf-8 -*-
import numpy as np
import pyNN.spiNNaker as spynnaker
import csv
import random

#
#  SpiNN-3 board configuration
#  @Before: Epochs              -> set to 100ms   
#           Spiking_time_space  -> shape:(100 dimensional 2d-array)
#           
#

def setupLayer_PN(time_space):
    '''
     PN ─┬─── pn_neuron_01
         ├─── pn_neuron_02
         ├─── pn_neuron_03
         ├─── ...
         └─── pn_neuron_100

     PN was used as input layer
    '''
    NUM_PN_CELLS = 100
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
    NUM_KC_CELLS  = 2000
    NEURON_PARAMS = {
                     'cm'        : 0.25,
                     'i_offset'  : 0.0,
                     'tau_m'     : 20.0,
                     'tau_refrac': 0.0,
                     'tau_syn_E' : 10.0,
                     'tau_syn_I' : 10.0,
                     'v_reset'   : -70.0,
                     'v_rest'    : -65.0,
                     'v_thresh'  : -64.0
                    }
    kc_population = spynnaker.Population(NUM_KC_CELLS,
                                         spynnaker.IF_curr_exp,
                                         NEURON_PARAMS,
                                         label='KC_population')
    return kc_population

def setupProjection_PN_KC(pn_population,kc_population):

    WEIGHT_PN_KC = 5
    DELAY_PN_KC  = 1.0
    NUM_KC_CELLS = 2000
    NUM_PN_CELLS = 100

    connectionList = list()                                        # Connection list between PN and KC
    for each_kc_cell in xrange(NUM_KC_CELLS):

        count          = 20
        selectedCells = random.sample(xrange(NUM_PN_CELLS),count) 

        for each_pn_cell in selectedCells:
            single_coonection = (each_pn_cell,each_kc_cell)
            connectionList.append(single_coonection)

    pnkcProjection = spynnaker.Projection(pn_population,
                                          kc_population,
                                          spynnaker.FromListConnector(connectionList),
                                          spynnaker.StaticSynapse(weight=WEIGHT_PN_KC, delay=DELAY_PN_KC))
    return pnkcProjection


def readData():
    spikeLists= []
    c = open("InputSpikingTime.csv", "rb")
    read = csv.reader(c)
    for line in read:
        spikeLists.append(map(float, line))
    return spikeLists



def retrieve_data(spikeData_original):

    TIME_SLOT = 100
    TOTAL_TIME= 1000000
    TIMER     = 0
    ROUND     = 0
    DIMENSION = len(spikeData_original)
    original  = [[] for i in range(DIMENSION+1)]
    simplified= []

    while(TIMER<TOTAL_TIME):
        temp_array = []
        for neuron_index in range(DIMENSION):
            temp = np.array(spikeData_original[neuron_index])
            count= ((temp>TIMER)&(temp<TIMER+TIME_SLOT)).sum()
            rate = float(count)/(float(TIME_SLOT)/1000)
            temp_array.append(rate)
        # -------------------------------------------------tmp数组表示这一次的原始数据
        original.append(temp_array)
        indices = np.argpartition(original[ROUND], 100)[:100]
        ROUND  += 1
        simplified[ROUND, :][indices] = original[ROUND, :][indices]

def save_data(src):
    LEN     = 2000
    csvFile = open("src.csv", "w")
    writer  = csv.writer(csvFile)

    for neuron_index in xrange(LEN):
        writer.writerow(src[neuron_index])
    csvFile.close()


def mapping_process():

    ##################
    SIM_TIME = 2500 ##
    ##################

    spynnaker.setup(timestep=1)
    spynnaker.set_number_of_neurons_per_core(spynnaker.IF_curr_exp, 50)
    time_space     = readData()

    pn_population  = setupLayer_PN(time_space)
    kc_population  = setupLayer_KC()
    kc_population.record(["spikes"])

    pn_kc_projection  = setupProjection_PN_KC(pn_population,kc_population)
    spynnaker.run(SIM_TIME)

    neo = kc_population.get_data(variables=["spikes"])
    spikeData_original= neo.segments[0].spiketrains
    spynnaker.end()
    return spikeData_original
