import numpy as np
import tensorflow as tf

def brackets(arr):
    res = "["
    if len(arr) > 1:
        for _ in range(arr[0]):
            res += brackets(arr[1:])
    else:
        for _ in range(arr[0]-1):
            res += " , "
    res += "]"
    return res

def b(arr):
    print(brackets(np.array(arr)))
