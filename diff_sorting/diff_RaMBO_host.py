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
    # with open('loma_code/diff_RaMBO.py') as f:
    #     _, lib = compiler.compile(f.read(),
    #                               target='c',
    #                               output_filename='_code/diff_RaMBO')
    with open('loma_code/diff_RaMBO_simplified.py') as f:
        _, lib = compiler.compile(f.read(),
                                  target='c',
                                  output_filename='_code/diff_RaMBO_simplified')
        
    py_score = [-0.4, -0.7, 0.3, 0.2, 0.5]
    py_dscore = [0.0, 0.0, 0.0, 0.0, 0.0]
    py_pos_score = [-0.4, -0.7, 0.3, 0.2, 0.5]
    py_rank = [0, 0, 0, 0, 0]
    py_pos_rank = [0, 0, 0, 0, 0]
    py_norm_rank = [0.0, 0.0, 0.0, 0.0, 0.0]
    py_pos_norm_rank = [0.0, 0.0, 0.0, 0.0, 0.0]
    py_d_norm_rank = [0.0, 0.0, 0.0, 0.0, 0.0]
    py_label = [0, 1, 0, 1, 0]

    score = (ctypes.c_float * len(py_score))(*py_score)
    d_score = (ctypes.c_float * len(py_dscore))(*py_dscore)
    pos_score = (ctypes.c_float * len(py_score))(*py_pos_score)
    rank = (ctypes.c_int * len(py_rank))(*py_rank)
    pos_rank = (ctypes.c_int * len(py_rank))(*py_pos_rank)
    norm_rank = (ctypes.c_float * len(py_norm_rank))(*py_norm_rank)
    pos_norm_rank = (ctypes.c_float * len(py_norm_rank))(*py_pos_norm_rank)
    d_norm_rank = (ctypes.c_float * len(py_norm_rank))(*py_d_norm_rank)
    label = (ctypes.c_int * len(py_label))(*py_label)
    loss_record = ctypes.c_float(0.0)

    size = ctypes.c_int(len(py_score))
    lambda_val = ctypes.c_float(4.0)

    # lib.call_RaMBO(score, d_score, pos_score, size, rank, pos_rank,
    #                norm_rank, pos_norm_rank, d_norm_rank, label, lambda_val)

    lib.call_RaMBO(score, d_score, size, rank, norm_rank, d_norm_rank, label, lambda_val, loss_record)

    # py_arr = [1.0, 2.0, 3.0, 4.0, 5.0]
    # arr = (ctypes.c_float * len(py_arr))(*py_arr)
    print(d_norm_rank[:5])
    print(d_score[:5])
    print(loss_record)
