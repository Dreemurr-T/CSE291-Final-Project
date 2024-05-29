# autodiff version of RaMBO: Blackbox differentiation for ranking

def ranker(arr : In[Array[float]], arr_size : In[int], reverse: In[int], rank: Out[Array[int]]):
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


def normalize_rank(rank : In[Array[int]], size : In[int], norm_rank : Out[Array[float]]):
    i : int = 0
    while (i < size, max_iter := 10000):
        norm_rank[i] = (rank[i] + 1.0) / (size + 1e-8)
        i = i + 1


def RaMBO_forward(score : In[Array[float]], size : In[int], rank : Out[Array[int]], norm_rank : Out[Array[float]]):
    ranker(score, size, 1, rank)
    normalize_rank(rank, size, norm_rank)

# not finished
def recall_loss(norm_rank : In[Array[float]], size : In[int], label : In[Array[int]], loss : Out[float]):
    i : int = 0
    while (i < size, max_iter := 10000):
        loss = loss + (label[i] - norm_rank[i]) * (label[i] - norm_rank[i])
        i = i + 1

d_rev_recall_loss = rev_diff(recall_loss)

# output gradient
def RaMBO_backward(score : In[Array[float]], score_prime : In[Array[float]], 
                   size : In[int], rank : In[Array[int]], rank_prime : In[Array[int]], norm_rank : In[Array[float]], norm_rank_prime : In[Array[float]], 
                   d_norm_rank : In[Array[float]], label : In[Array[int]], d_label : In[Array[int]], lambda_val : In[float], gradient : Out[Array[float]]):
    
    i : int
    RaMBO_forward(score, size, rank, norm_rank)

    d_size : int
    loss : float
    d_rev_recall_loss(norm_rank, d_norm_rank, size, d_size, label, d_label, loss)
    
    while (i < size, max_iter := 10000):
        score_prime[i] = score[i] + lambda_val * d_norm_rank[i]

    RaMBO_forward(score_prime, size, rank_prime, norm_rank_prime)

    while (i < size, max_iter := 10000):
        gradient[i] = -(norm_rank_prime[i] - norm_rank[i]) / (lambda_val + 1e-8)
        i = i + 1