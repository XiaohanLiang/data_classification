#
# @Function: This function was used to analyse the resulting data
#            1. Transfer spiking time into spiking rate
#            2. Deploy 5% inhibition method 
#            3. Calculate the Euclidean-distance to find 5 nearest neighbor
#
# @Mention: This version will not have verification step, 
#           we just list the 5-nearest neighbor directly
#

import numpy as np
from calculate_cosine_distance import *
import heapq

params = eval(open("settings.txt").read())
TIME_SLOT           = params['TIME_SLOT']
KC_CELL_AMOUNT      = params['NUM_KC_CELLS']
HASH_LENGTH         = params['HASH_LENGTH']
DATA_AMOUNT         = params['DATA_AMOUNT']
NN_LIST_LENGTH      = params['LIST_LENGTH']

#
# @Function: first_part -> transfer raw spiking time into spiking rate
#            second_part-> deploy 5% inhibition to spiking rate to get hashed
#
# @Params: spiking_space -> raw data retrieved from machine
#
# @Return: simplified -> shape=(graph_count,kc_count)
#
def get_count(spiking_space):

    spiking_count = np.zeros((DATA_AMOUNT,KC_CELL_AMOUNT))

    for graph_index in xrange(DATA_AMOUNT):
        begin_time = graph_index*TIME_SLOT
        end_time   = begin_time+TIME_SLOT
        for neuron_index in xrange(KC_CELL_AMOUNT): 
            #spiking_record  = np.array(spiking_space[neuron_index])
            spiking_record  = spiking_space[neuron_index]
            count = len([element for element in spiking_record if element>begin_time and element<end_time])
            spiking_count[graph_index][neuron_index] = count
    
    np.savetxt('./retrived_data/spiking_count.txt',spiking_count,fmt='%d',delimiter=',',newline='\n')

    simplified   =  np.zeros((DATA_AMOUNT,KC_CELL_AMOUNT))
        
    for graph_index in xrange(DATA_AMOUNT):
        indices = np.argpartition(spiking_count[graph_index], -HASH_LENGTH)[-HASH_LENGTH:]
        simplified[graph_index][indices] = spiking_count[graph_index][indices]

    np.savetxt('./retrived_data/simplified_count.txt',simplified,fmt='%d',delimiter=',',newline='\n')

    return simplified

#
# @Function: Find 5 nearest neighbors of each graph
#
# @Params: simplified or hashed spiking rate
#
# @Returns: NN_List -> shape(graph_count,5)
#           for each graph this matrix contains its 5 nearest neighbors
#
def get_nearest_neighbor(simplified):

    distance_list = [[] for i in xrange(DATA_AMOUNT)]
    NN_list = [[] for i in xrange(DATA_AMOUNT)]
    
    for graph_a in xrange(DATA_AMOUNT):
        
        for graph_b in xrange(DATA_AMOUNT):
            distance = np.linalg.norm((simplified[graph_a]-simplified[graph_b]),ord=2)
            distance_list[graph_a].append((distance,graph_b))
        
        NN_list[graph_a]  =  heapq.nsmallest(NN_LIST_LENGTH,distance_list[graph_a])
        NN_list[graph_a]  = np.array([vals[1] for vals in NN_list[graph_a]])
    
    return NN_list

