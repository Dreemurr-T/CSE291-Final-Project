import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler
import ctypes
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    with open('loma_code/RaMBO.py') as f:
        _, lib = compiler.compile(f.read(),
                                  target = 'c',
                                  output_filename = '_code/RaMBO')

    # py_arr = [1.0, 2.0, 3.0, 4.0, 5.0]
    # arr = (ctypes.c_float * len(py_arr))(*py_arr)
    # assert abs(lib.sum_array(arr, len(py_arr)) - 15.0) < 1e-6