# autodiff version of RaMBO: Blackbox differentiation for ranking
class Rank:
    score : Array[float, 10000]
    rank: Array[int, 10000]
    norm_rank : Array[float, 10000]
    size : int
#     _lambda : float

def ranker(r : In[Rank], reverse : In[int]) -> Rank:
    i : int = 0
    j : int = 0
    cnt : int = 0
    while (i < r.size, max_iter := 10000):
        cnt = 0
        j = 0
        while (j < r.size, max_iter := 10000):
            if reverse == 1:
                if r.score[j] > r.score[i]:
                    cnt = cnt + 1
            else:
                if r.score[j] < r.score[i]:
                    cnt = cnt + 1
            j = j + 1
        r.score[i] = cnt
        i = i + 1

# d_argsort_bubble_sort = fwd_diff(argsort_bubble_sort)

def normalize_rank(r : In[Rank]) -> Rank:
    i : int = 0
    while (i < r.size, max_iter := 10000):
        r.norm_rank[i] = (r.rank[i] + 1) / (r.size + 1e-8)
        i = i + 1
    
    return r

# d_normaliza_rank = fwd_diff(normalize_rank)

def RaMBO_forward(r : In[Rank]) -> Rank:
    r = ranker(r, 0)
    r = normalize_rank(r)

    # Loss function here

    return r

d_fwd_RaMBO_forward = fwd_diff(RaMBO_forward)

def fwd_diff_RaMBO_backward(r : In[Rank], lambda_val : In[float]) -> Array[Diff[float], 10000]:
    d_r : Diff[Rank]
    i : int = 0
    d_r.size = r.size

    while (i < r.size, max_iter := 10000):
        d_r.score[i].val = r.score[i]
        # d_r.score[i].dval = 0.0
        # d_r.rank[i] = i
        # d_r.norm_rank[i].val = 0.0
        # d_r.norm_rank[i].dval = 0.0

        i = i + 1

    d_r = d_fwd_RaMBO_forward(d_r)
    
    while (i < r.size, max_iter := 10000):
        r.score[i] = r.score[i] + lambda_val * d_r.norm_rank[i].dval
        i = i + 1

    r = RaMBO_forward(r)

    while (i < d_r.size, max_iter := 10000):
        d_r.score[i].dval = -(d_r.norm_rank[i].val - r.norm_rank[i]) / (lambda_val + 1e-8)
        i = i + 1
    
    return d_r.score