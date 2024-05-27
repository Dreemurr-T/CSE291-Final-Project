# input an array
# output the sorted array

# RaMBO: Blackbox differentiation for ranking
class Rank:
    arr : Array[float]
    rank : Array[float]
    arr_size : int
    # _lambda : float

def normalize_rank(rank : In[Array[int]], arr_size : In[int], normalizaed_rank: Out[Array[float]]):
    i : int = 0
    while (i < arr_size, max_iter := 1000):
        normalizaed_rank[i] = (rank[i] + 1) / arr_size
        i = i + 1

def RaMBO_forward(arr : In[Array[float]], arr_size : In[int]) -> Rank:
    out_f : Rank
    out_f.arr = arr
    out_f.arr_size = arr_size

    rank : Array[int]
    argsort_bubble_sort(arr, arr_size, 0, rank)
    normalize_rank(rank, arr_size, out_f.rank)

    # Loss function here ?

    return out_f

d_RaMBO_forward = fwd_diff(RaMBO_forward)

def RaMBO_backward(arr : In[Array[float]], arr_size : In[int], lamb : In[float], g_out : Out[Array[float]]):
    d_arr : Diff[Array[float]]
    d_arr.val = arr
    d_arr.dval = 0

    d_rank : Diff[Rank] = d_RaMBO_forward(arr, d_arr, arr_size)

    arr_prime : Array[float]
    while (i < arr_size, max_iter := 10000):
        arr_prime[i] = arr[i] + lamb * d_rank.rank[i].dval
        i = i + 1
    rank_prime : Array[int]
    rank_prime_normalized : Array[float]

    argsort_bubble_sort(arr_prime, arr_size, 0, rank_prime)
    normalize_rank(rank_prime, arr_size, rank_prime_normalized)

    while (i < arr_size, max_iter := 10000):
        g_out[i] = -(d_rank.rank[i].val - rank_prime_normalized[i]) / lamb
        i = i + 1

d_RaMBO_backward = rev_diff(RaMBO_backward)

# use loma to implement a typical sorting function and ranking function
def bubble_sort(arr : In[Array[float]], arr_size : In[int], reverse: In[int], sorted_arr: Out[Array[float]]):
    i : int = 0
    j : int = 0
    tmp : float = 0.0
    swapped: int = 0
    while (i < arr_size, max_iter := 10000):
        sorted_arr[i] = arr[i]
        i = i + 1
    
    i = 0
    while (i < arr_size, max_iter := 10000):
        j = 0
        swapped = 0
        while (j < arr_size - i - 1, max_iter := 10000):
            if reverse == 1:
                if sorted_arr[j] < sorted_arr[j + 1]:
                    tmp = sorted_arr[j]
                    sorted_arr[j] = sorted_arr[j + 1]
                    sorted_arr[j + 1] = tmp
                    swapped = 1
            else:
                if sorted_arr[j] > sorted_arr[j + 1]:
                    tmp = sorted_arr[j]
                    sorted_arr[j] = sorted_arr[j + 1]
                    sorted_arr[j + 1] = tmp
                    swapped = 1
            j = j + 1
        i = i + 1
        if swapped == 0:
            i = arr_size

def argsort_bubble_sort(arr : In[Array[float]], arr_size : In[int], reverse: In[int], rank: Out[Array[int]]):
    i : int = 0
    j : int = 0
    tmp_index: int = 0
    swapped: int = 0

    while (i < arr_size, max_iter := 10000):
        rank[i] = i
        i = i + 1
    i = 0
    while (i < arr_size, max_iter := 10000):
        j = 0
        swapped = 0
        while (j < arr_size - i - 1, max_iter := 10000):
            if reverse == 1:
                if arr[rank[j]] < arr[rank[j + 1]]:
                    tmp_index = rank[j]
                    rank[j] = rank[j + 1]
                    rank[j + 1] = tmp_index
                    swapped = 1
            else:
                if arr[rank[j]] > arr[rank[j + 1]]:
                    tmp_index = rank[j]
                    rank[j] = rank[j + 1]
                    rank[j + 1] = tmp_index
                    swapped = 1
            j = j + 1
        i = i + 1
        if swapped == 0:
            i = arr_size
    
