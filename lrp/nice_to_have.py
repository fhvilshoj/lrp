import numpy as np
import tensorflow as tf

g = tf.Graph()
g = g.as_default()

s = tf.Session()

def brackets(arr, default_element):
    res = "["
    if len(arr) > 1:
        for i in range(arr[0]):
            res += brackets(arr[1:], default_element)
            if i < arr[0] - 1:
                res += ","
    else:
        for _ in range(arr[0]-1):
            res += default_element + ", "
            res += default_element
    res += "]"
    return res

def b(arr, default_element=""):
    if not isinstance(default_element, str):
        default_element = str(default_element)
    print(brackets(np.array(arr), default_element))


def print_array_from_paste_book():
    with open('playground/paste_book.txt') as f:
        arr = []
        for line in f:
            if len(line) == 0:
                continue
            arr.append(list(map(lambda str: float(str.replace(",", ".")), line.split("	"))))

        print(arr)
