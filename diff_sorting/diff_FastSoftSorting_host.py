import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler
import ctypes
import numpy as np
import matplotlib.pyplot as plt


def test_fast_soft_sorting(arr, structs, lib):
    size = len(arr)
    reverse = 0
    sorted_arr = np.zeros(size, dtype=np.float32)
    permutation = np.zeros(size, dtype=np.float32)


    regularization_strength = 1.0

    _dfloat = structs['_dfloat']
    x = _dfloat(0.3, 0.4)
    i = 1
    j = _dfloat(3.5, 0.5)
    py_y = [_dfloat(0, 0)] * 7
    y = (_dfloat * len(py_y))(*py_y)

    lib.d_fast_soft_sorting(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, reverse, regularization_strength, sorted_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), permutation.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    py_sorted_arr = np.sort(arr)
    print(py_sorted_arr)
    print(sorted_arr)
    
    # for i in range(size):
    #     assert sorted_arr[i] == py_sorted_arr[i]


if __name__ == '__main__':
    with open('loma_code/FastSoftSorting.py') as f:
        structs, lib = compiler.compile(f.read(),
                                  target='c',
                                  output_filename='_code/fast_soft_sorting')
    

    arr = np.array([64.2, 34.5, 25.5, 12.22, 11.0, 90.1]).astype(np.float32)
    test_fast_soft_sorting(arr, lib)
