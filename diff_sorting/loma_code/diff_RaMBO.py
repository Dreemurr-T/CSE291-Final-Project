# autodiff version of RaMBO: Blackbox differentiation for ranking
class Rank:
    arr : Array[float, 1000]
    rank: Array[int, 1000]
    normalized_rank : Array[float, 1000]
    arr_size : int
#     _lambda : float

def ranker(arr : In[Array[float, 1000]], arr_size : In[int], reverse: In[int], rank: Out[Array[int, 1000]]):
    i : int = 0
    j : int = 0
    cnt : int = 0
    while (i < arr_size, max_iter := 10000):
        cnt = 0
        j = 0
        while (j < arr_size, max_iter := 10000):
            if reverse == 1:
                if arr[j] > arr[i]:
                    cnt = cnt + 1
            else:
                if arr[j] < arr[i]:
                    cnt = cnt + 1
            j = j + 1
        rank[i] = cnt
        i = i + 1

# d_argsort_bubble_sort = fwd_diff(argsort_bubble_sort)

def normalize_rank(rank : In[Array[int, 1000]], arr_size : In[int], normalizaed_rank: Out[Array[float, 1000]]):
    i : int = 0
    while (i < arr_size, max_iter := 1000):
        normalizaed_rank[i] = (rank[i] + 1) / (arr_size + 1e-8)
        i = i + 1

# d_normaliza_rank = fwd_diff(normalize_rank)

def RaMBO_forward(r : In[Rank]) -> Rank:
    ranker(r.arr, r.arr_size, 0, r.rank)
    normalize_rank(r.rank, r.arr_size, r.normalized_rank)

    # Loss function here ?

    return r

d_RaMBO_forward = fwd_diff(RaMBO_forward)

def RaMBO_backward(r : In[Rank], lambda_val : In[float]) -> Array[float, 1000]:
    d_r : Diff[Rank]
    i : int = 0

    while (i < r.arr_size, max_iter := 10000):
        d_r.arr[i].val = r.arr[i]
        d_r.arr[i].dval = 0.0
        d_r.rank[i] = i
        d_r.normalized_rank[i].val = 0.0
        d_r.normalized_rank[i].dval = 0.0

        i = i + 1

    d_r.arr_size = r.arr_size
    
    d_r = d_RaMBO_forward(d_r)
    
    while (i < r.arr_size, max_iter := 10000):
        r.arr[i] = r.arr[i] + lambda_val * d_r.normalized_rank[i].dval
        i = i + 1

    r = RaMBO_forward(r)

    gradient_out : Array[float, 1000]

    while (i < d_r.arr_size, max_iter := 10000):
        gradient_out[i] = -(d_r.normalized_rank[i].val - r.normalized_rank[i]) / (lambda_val + 1e-8)
        i = i + 1
    
    return gradient_out