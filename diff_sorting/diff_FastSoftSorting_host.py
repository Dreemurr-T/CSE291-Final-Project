import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler
import ctypes
import numpy as np
import matplotlib.pyplot as plt


def test_fast_soft_sorting(arr, lib):
    size = len(arr)
    reverse = 0
    sorted_arr = np.zeros(size, dtype=np.float32)
    # permutation = np.zeros(size, dtype=np.float32)

    arr = (ctypes.c_float * size)(*arr)
    py_darr = [0]*size
    _darr = (ctypes.c_float * size)(*py_darr)

    _dsize = ctypes.c_int(0)
    _dreverse = ctypes.c_int(0)

    regularization_strength = 1.0
    _dregularization_strength = ctypes.c_float(0)

    py_dsorted_arr = [0]*size
    _dsorted_arr = (ctypes.c_float * size)(*py_dsorted_arr)

    py_sorted_arr = np.sort(arr)

    lib.d_fast_soft_sorting(arr, _darr, size, ctypes.byref(_dsize), reverse, ctypes.byref(_dreverse), regularization_strength, ctypes.byref(_dregularization_strength), _dsorted_arr)
    
    print(py_sorted_arr)
    print(_darr[:size])
    
    # for i in range(size):
    #     assert sorted_arr[i] == py_sorted_arr[i]


if __name__ == '__main__':
    with open('loma_code/FastSoftSorting.py') as f:
        structs, lib = compiler.compile(f.read(),
                                  target='c',
                                  output_filename='_code/fast_soft_sorting')
    

    arr = np.array([64.2, 34.5, 25.5, 12.22, 11.0, 90.1]).astype(np.float32)
    test_fast_soft_sorting(arr, lib)
