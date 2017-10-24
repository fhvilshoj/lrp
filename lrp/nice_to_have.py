import numpy as np
import tensorflow as tf

g = tf.Graph()
g = g.as_default()

s = tf.Session()

def brackets(arr):
    res = "["
    if len(arr) > 1:
        for i in range(arr[0]):
            res += brackets(arr[1:])
            if i < arr[0] - 1:
                res += ","
    else:
        for _ in range(arr[0]-1):
            res += " , "
    res += "]"
    return res

def b(arr):
    print(brackets(np.array(arr)))


def print_array_from_paste_book():
    with open('playground/paste_book.txt') as f:
        arr = []
        for line in f:
            if len(line) == 0:
                continue
            arr.append(list(map(lambda str: float(str.replace(",", ".")), line.split("	"))))

        print(arr)
