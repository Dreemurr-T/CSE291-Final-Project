import torch
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import compiler
import ctypes
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Compile loma code
with open('loma_code/FastSoftSorting.py') as f:
    structs, lib = compiler.compile(f.read(),
                                target='c',
                                output_filename='_code/fast_soft_sorting')


class DiffSortFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, arr, reverse, regularization_strength):
        ctx.save_for_backward(arr)

        if arr.device.type == 'cuda':
            arr = arr.cpu().numpy()
        else:
            arr = arr.numpy()
     
        ctx.reverse = reverse
        ctx.regularization_strength = regularization_strength
        size = arr.shape[0]
        
        arr = (ctypes.c_float * size)(*arr)

        sorted_arr = np.zeros(size, dtype=np.float32)
        sorted_arr = (ctypes.c_float * size)(*sorted_arr)

        lib.soft_sort(arr, size, reverse, regularization_strength, sorted_arr)
        sorted_arr = torch.tensor(sorted_arr[:size], dtype=torch.float32).to(device)
        return sorted_arr
    
    @staticmethod
    def backward(ctx, grad_output):
        arr = ctx.saved_tensors
        arr = arr[0]

        if arr.device.type == 'cuda':
            arr = arr.cpu().numpy()
            grad_output = grad_output.cpu().numpy()
        else:
            arr = arr.numpy()
            grad_output = grad_output.numpy()

        size = arr.shape[0]

        reverse = ctx.reverse
        regularization_strength = ctx.regularization_strength

        arr = (ctypes.c_float * size)(*arr)
        py_darr = [0]*size
        _darr = (ctypes.c_float * size)(*py_darr)

        _dsize = ctypes.c_int(0)
        _dreverse = ctypes.c_int(0)
        _dregularization_strength = ctypes.c_float(0)

        _dsorted_arr = (ctypes.c_float * size)(*grad_output)

        lib.d_fast_soft_sorting(arr, _darr, size, ctypes.byref(_dsize), reverse, ctypes.byref(_dreverse), regularization_strength, ctypes.byref(_dregularization_strength), _dsorted_arr)
        _darr = torch.tensor(_darr[:size], dtype=torch.float32).to(device)
        isnan = torch.isnan(_darr)
        has_nan = torch.any(isnan)
        if has_nan:
            _darr[isnan] = 0.1
        return _darr, None, None