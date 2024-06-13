import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler
import ctypes
import numpy as np
import matplotlib.pyplot as plt



def test_ascending_bubble_sort(arr, lib):
    size = len(arr)
    reverse = 0
    sorted_arr = np.zeros(size, dtype=np.float32)
    lib.bubble_sort(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, reverse, sorted_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    py_sorted_arr = np.sort(arr)
    for i in range(size):
        assert sorted_arr[i] == py_sorted_arr[i]

def test_descending_bubble_sort(arr, lib):
    size = len(arr)
    reverse = 1
    sorted_arr = np.zeros(size, dtype=np.float32)
    lib.bubble_sort(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, reverse, sorted_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    py_sorted_arr = np.sort(arr)[::-1]
    for i in range(size):
        assert sorted_arr[i] == py_sorted_arr[i]

def test_argsort_increasing_order(arr, lib):
    size = len(arr)
    reverse = 0
    rank = np.zeros(size, dtype=np.int32)
    lib.argsort_bubble_sort(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, reverse, rank.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    py_rank = np.argsort(arr)
    for i in range(size):
        assert rank[i] == py_rank[i]

def test_argsort_decreasing_order(arr, lib):
    size = len(arr)
    reverse = 1
    rank = np.zeros(size, dtype=np.int32)
    lib.argsort_bubble_sort(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, reverse, rank.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    py_rank = np.argsort(arr)[::-1]
    for i in range(size):
        assert rank[i] == py_rank[i]

def test_stable_argsort_increasing_order(arr, lib):
    size = len(arr)
    reverse = 0
    rank = np.zeros(size, dtype=np.int32)
    lib.argsort_bubble_sort(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, reverse, rank.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    py_rank = np.argsort(arr, kind='stable')
    for i in range(size):
        assert rank[i] == py_rank[i]

def test_stable_argsort_decreasing_order(arr, lib):
    size = len(arr)
    reverse = 1
    rank = np.zeros(size, dtype=np.int32)
    lib.argsort_bubble_sort(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, reverse, rank.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    py_rank = size - 1 - np.argsort(arr[::-1], kind='stable')[::-1]
    for i in range(size):
        assert rank[i] == py_rank[i]

def test_ranker_ascending_order(arr, lib):
    size = len(arr)
    rank = np.zeros(size, dtype=np.int32)
    lib.ranker(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, 0, rank.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    ranks = []
    for i in range(len(arr)):
        count = sum(x < arr[i] for x in arr)
        ranks.append(count)
    for i in range(size):
        assert rank[i] == ranks[i]

def test_ranker_descending_order(arr, lib):
    size = len(arr)
    rank = np.zeros(size, dtype=np.int32)
    lib.ranker(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size, 1, rank.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    ranks = []
    for i in range(len(arr)):
        count = sum(x > arr[i] for x in arr)
        ranks.append(count)
    for i in range(size):
        assert rank[i] == ranks[i]


if __name__ == '__main__':
    with open('loma_code/differentiable_sorting.py') as f:
        structs, lib = compiler.compile(f.read(),
                                  target = 'c',
                                  output_filename = '_code/differentiable_sorting')

    arr = np.array([64.2, 34.5, 25.5, 12.22, 11.0, 90.1]).astype(np.float32)
    test_ascending_bubble_sort(arr, lib)
    test_descending_bubble_sort(arr, lib)
    test_argsort_increasing_order(arr, lib)
    test_argsort_decreasing_order(arr, lib)

    arr = np.array([-0.3, 0.8, -5.0, 3.0, 1.0]).astype(np.float32)
    test_stable_argsort_increasing_order(arr, lib)

    arr = np.array([3.0, 1.0, 2.0, 3.0, 3.0]).astype(np.float32)
    test_stable_argsort_increasing_order(arr, lib)
    test_stable_argsort_decreasing_order(arr, lib)
    test_ranker_ascending_order(arr, lib)
    test_ranker_descending_order(arr, lib)
    print('All tests passed')