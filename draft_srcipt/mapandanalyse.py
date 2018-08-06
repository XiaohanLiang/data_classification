# -*- coding:utf-8 -*-
import numpy as np
import pyNN.spiNNaker as spynnaker
import csv
import random
import sys
import time
import analyse as ana
'''
    板子设置
'''
# 因为spike时间已经扩大为原先的十倍
# 因此现在运行时间也变成10000ms

SIM_TIME = 0

def operation_time(process_name,begin,end):

    instream_type = 'a'
    operation_name= process_name

    f = open("processing_time_record.txt",instream_type)
    f.write(operation_name+" takes time: "+str(end-begin)+"\n")
    f.close()

    return

def setupLayer_PN(time_space):
    '''
     PN ─┬─── pn_neuron_01
         ├─── pn_neuron_02
         ├─── pn_neuron_03
         ├─── ...
         └─── pn_neuron_784
    '''
    NUM_PN_CELLS = 784
    '''
        784只PN神经元放在
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
               Each KC neuron map to around ~6 PN_cells
               which was chosen randomly from 784 of all
               besides, by the property of SpiNNaker Board
               (each core 256 neurons MAX)
               2000 KC_neurons will spreads to around ~10 cores
    '''
    NUM_KC_CELLS  = 2000
    NEURON_PARAMS = {'cm': 0.25,
                     'i_offset': 0.0,
                     'tau_m': 20.0,
                     'tau_refrac': 0.0,
                     'tau_syn_E': 10.0,
                     'tau_syn_I': 10.0,
                     'v_reset': -70.0,
                     'v_rest': -65.0,
                     'v_thresh': -64.0
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
    NUM_PN_CELLS = 784


    connectionList = list()                              # Build up a connection list between PN and KC
    for each_kc_cell in xrange(NUM_KC_CELLS):

        count  = random.randint(5,7)                   # How many pn_cells will connect to this kc_cell
        selectedCells = random.sample(xrange(NUM_PN_CELLS),count) # Index of randomly selected PN cells

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
    c = open("GNG-optimum-VR-set.csv", "rb")
    read = csv.reader(c)
    for line in read:
        spikeLists.append(map(float, line))
    return spikeLists

def save_data(src):
    LEN     = 2000
    csvFile = open("src.csv", "w")
    writer  = csv.writer(csvFile)

    for neuron_index in xrange(LEN):
        writer.writerow(src[neuron_index])
    csvFile.close()


def mapping_process():

    assert SIM_TIME>0
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


if __name__=='__main__':

    NUMBER_OF_DATA = int(sys.argv[1])
    SIM_TIME = NUMBER_OF_DATA*50   # 暂时默认expose_time=50
    begin = time.time()
    spikeData_original = mapping_process()
    end   = time.time()
    operation_time("***Whole process of Map.py",begin=begin,end=end)
    ana.analyse(spikeData_original)