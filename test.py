#
# This file was used to test for various purpose
# Here I use it to test csv reading
#
import numpy as np
dir_to_csv = "./GNG-optimum-VR-set.csv" 
def reader():
    obj = np.loadtxt(dir_to_csv,delimiter=',')
    print(obj)

reader()
