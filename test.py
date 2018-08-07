#
# This file was used to test for various purpose
# Here I use it to test csv reading
#
import numpy as np
a = [1,1,1]
a = np.array(a)
b = [2,2,2]
b = np.array(b)
c = np.linalg.norm((a-b),ord=2)
print(c)
