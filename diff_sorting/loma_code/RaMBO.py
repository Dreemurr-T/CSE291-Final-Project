# non-autodiff version of RaMBO: Blackbox differentiation for ranking
class Rank:
    score : Array[float]
    rank: Array[int]
    norm_rank : Array[float]
    gradient : Array[float]
    size : int

def ranker(r : In[Rank], reverse: In[int]) -> Rank:
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
        r.rank[i] = cnt
        i = i + 1
    
    return r

def normalize_rank(r : In[Rank]) -> Rank:
    i : int = 0
    while (i < r.size, max_iter := 1000):
        r.norm_rank[i] = (r.rank[i] + 1) / (r.size + 1e-8)
        i = i + 1
    
    return r

def RaMBO_forward(r : In[Rank]) -> Rank:
    r = ranker(r, 0)
    r = normalize_rank(r)

    # Can put loss function here
    return r

def RaMBO_backward(r : In[Rank], tmp_r : In[Rank], lambda_val : In[float], loss : In[Array[float]]) -> Rank:
    i : int = 0
    
    while (i < r.size, max_iter := 10000):
        tmp_r.score[i] = r.score[i] + lambda_val * loss[i]
        i = i + 1

    tmp_r = RaMBO_forward(tmp_r)

    while (i < r.size, max_iter := 10000):
        r.gradient[i] = -(r.norm_rank[i] - tmp_r.norm_rank[i]) / (lambda_val + 1e-8)
        i = i + 1
    
    return r