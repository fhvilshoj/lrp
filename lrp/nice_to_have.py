import numpy as np
import tensorflow as tf
import subprocess
from sys import platform

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
        for _ in range(arr[0] - 1):
            res += default_element + ", "
        res += default_element
    res += "]"
    return res


def br(arr, default_element=""):
    if not isinstance(default_element, str):
        default_element = str(default_element)
    print(brackets(np.array(arr), default_element))


def get_text_to_copy_to_sheets(arr):
    rank = np.ndim(arr)
    if rank == 1:
        s = ""
        for i in arr:
            s += str(i).replace(".", ",") + "	"
        print(s)
        return
    else:
        for i in arr:
            get_text_to_copy_to_sheets(i)
    print("\r\n")


def pr(arr):
    if not type(arr).__module__ == np.__name__:
        arr = np.array(arr)
    get_text_to_copy_to_sheets(arr)


def copy2clip(txt):
    if 'linux' in platform:
        cmd='echo ' + txt.strip() + '| xclip -selection clipboard'
        return subprocess.check_call(cmd, shell=True)
    else:
        print(txt)

def print_array_from_paste_book():
    arr = []
    with open('playground/paste_book.txt') as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            arr.append(
                list(map(lambda str: float(str.replace(",", ".")) if "," in str else int(str), line.split("	"))))

    copy2clip(str(arr).replace('],', '],\n'))
