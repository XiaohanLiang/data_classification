# -*- coding:utf-8 -*-
import numpy as np
'''
    将所有的784维的mnist数据
    生成784维的spike time
    保存
'''
# 这里现在有100个数据的spike信息
# 时间已经扩大十倍     -> 修改了expose_time + time_interval
# 处理因为频繁spike而产生的丢包问题
def transferData():

    NUMBER_OF_DATA = 50
    DIMENSION      = 784
    EXPOSE_TIME    = 50
    DATA_PATH      = "./data/mnist/mnist10k.txt"

    eigen_space    = np.zeros((NUMBER_OF_DATA, DIMENSION))
    with open(DATA_PATH) as f:
        for line_num, line in enumerate(f):
            if(line_num==NUMBER_OF_DATA): break # 只读取20个
            cols = line.strip().split(",")
            eigen_space[line_num, :] = map(float, cols)
    # ----------------------------------------------------rate数据已读取
    time_space     = [[0.0] for i in range(DIMENSION+1)]
    for data_count in range(NUMBER_OF_DATA):
        for Neuron_index in range(DIMENSION):
            if(eigen_space[data_count][Neuron_index]!=0):
                time_interval = (1.0/eigen_space[data_count][Neuron_index])*5
                # print(time_interval)
                while(time_space[Neuron_index][-1] < (data_count+1)*EXPOSE_TIME):
                    time_space[Neuron_index].append(round(time_space[Neuron_index][-1]+time_interval,2))
    # -----------------------------------------------------时间数据已经生成
    return time_space