# -*- coding:utf-8 -*-
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
import csv
import heapq
import random

EXPOSE_TIME         = 50
TOTAL_TIME          = 2500
KC_CELL_COUNT       = 2000
HASH_LENGTH         = 32
DATA_AMOUNT         = 50
MAP_LIST_LENGTH     = 10
RANDOM_TEST_AMOUNT  = DATA_AMOUNT

def readData():
    spikeLists_str= []
    spikeLists = []
    c = open("src.csv", "rb")
    read = csv.reader(c)
    for line in read:
        spikeLists_str.append(line)

    for one_row in spikeLists_str:
        row_digit = []
        for one_element in one_row:
            one_element = one_element[:-3]
            row_digit.append(one_element)
            row_digit=map(float,row_digit)
        spikeLists.append(row_digit)
        '''
            spikeLists[0] -> 第0个神经元所有的spike时间
        '''
    return spikeLists

def get_count(spikeLists):
    spiking_counter = []

    for slot in range(DATA_AMOUNT):
        one_expose = []
        for neuron in spikeLists:
            temp  = np.array(neuron)
            count = ((temp>slot*EXPOSE_TIME)&(temp<(slot+1)*EXPOSE_TIME)).sum()
            one_expose.append(count)

        if(sum(one_expose)==0):
            print("{}号位出现全零情况".format(slot))
        spiking_counter.append(one_expose)
        '''
            len(spiking_counter) -> 一共有多少个数据
            spiking_counter[0] -> 第0个数据下2000个神经元spike次数统计  
                                  (时间:0ms~50ms下的次数统计)
        '''
    return spiking_counter

def get_hash(spiking_counter):

    assert DATA_AMOUNT == len(spiking_counter)
    simplified_KCRate   =  np.zeros((DATA_AMOUNT,KC_CELL_COUNT))

    for a_data in range(DATA_AMOUNT):
        r = np.array(spiking_counter[a_data])
        indices = np.argpartition(r, -HASH_LENGTH)[-HASH_LENGTH:]
        simplified_KCRate[a_data][indices] = r[indices]
    '''
        前5%已提取并保存至simplified
    '''
    return simplified_KCRate

def MAP_calculation(orignal,simplified):

    MAP = []
    assert DATA_AMOUNT == len(simplified)
    index_set          = random.sample(xrange(DATA_AMOUNT), RANDOM_TEST_AMOUNT)

    for random_index in index_set:
        original_distance_measure   = []
        simplified_distance_measure = []

        for index_1 in xrange(DATA_AMOUNT):
            # 计算这个数据与其他所有数据之间的距离

            if(random_index==index_1): continue

            original_distance = np.linalg.norm((orignal[index_1],orignal[random_index]),ord=2)
            original_distance_measure.append((original_distance,index_1))

            simplified_distance = np.linalg.norm((simplified[index_1],simplified[random_index]),ord=2)
            simplified_distance_measure.append((simplified_distance,index_1))
            '''
                original_distance_measure   -> 这个数据与其他点之间的真实距离
                simplified_distance_measure -> 这个数据与其他点的预测距离
            '''

        original_NearestNeighbor   = heapq.nsmallest(MAP_LIST_LENGTH,
                                                     original_distance_measure)
        original_NearestNeighbor   = set([vals[1] for vals in original_NearestNeighbor])
        # print(original_NearestNeighbor)

        simplified_NearestNeighbor = heapq.nsmallest(MAP_LIST_LENGTH,
                                                     simplified_distance_measure)
        simplified_NearestNeighbor = set([vals[1] for vals in simplified_NearestNeighbor])
        # print(simplified_NearestNeighbor)

        correct_prediction = 0
        stats_array        = []

        if(index_1>=48):
            print("简化的最临近list")
            for item in simplified_NearestNeighbor:
                print(item)
            print("原始的最临近list")
            for item in original_NearestNeighbor:
                print(item)
            return


        for index,prediction in enumerate(simplified_NearestNeighbor):
            if(prediction in original_NearestNeighbor):
                correct_prediction+=1
                stats_array.append((correct_prediction)/(index+1))

        if(len(stats_array)==0):
            print("出现了全空数组: 序号为{}".format(random_index))
            stats = 0
        else:
            stats = np.mean(stats_array)
        MAP.append(stats)
        '''
            NearestNeighbor -> 邻近点的集合
            stats_array -> 执行MAP时的计算
            stats  -> 这一个数据的MAP
            MAP  -> 所有数据的MAP
        '''

    overall_score = np.mean(MAP)
    return overall_score


def analyse(spikeLists):
    KC_spikingRate = get_count(spikeLists)
    simplified_KCRate = get_hash(KC_spikingRate)
    overall_score = MAP_calculation(KC_spikingRate,simplified_KCRate)
    # print("################################################ - Congfig")
    # print("ExposeTime:{}".format(EXPOSE_TIME))
    # print("BoardOperationTime:{}".format(TOTAL_TIME))
    # print("NumofKC:{}".format(KC_CELL_COUNT))
    # print("➡️ Hash_Length ⬅️  :{}".format(HASH_LENGTH))
    # print("Data Amount in original distance measure:{}".format(DATA_AMOUNT))
    # print("Num of Nearest Neighbors in MAP computing:{}".format(MAP_LIST_LENGTH))
    # print("How many samples were used in MAP computing:{}".format(RANDOM_TEST_AMOUNT))
    # print("############################################################")
    print("Overall MAP : {}".format(overall_score))
