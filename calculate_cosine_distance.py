
#
# @Function: This function make a way to calculate
#            the cosine distance of two vectors 
#
import math

def cosine_distance(a,b):
    assert(len(a)==len(b))
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1,b1 in zip(a,b):
        part_up += a1*b1
        a_sq = a1**2
        b_sq = b1**2

    part_down = math.sqrt(a_sq*b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up/part_down

