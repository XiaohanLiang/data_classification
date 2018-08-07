#
# This file was used to test for various purpose
# Here I use it to test csv reading
#
import numpy as np
a = np.zeros((10,10))
np.savetxt("./test.txt",a,delimiter=',',newline='\n')
