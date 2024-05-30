# autodiff version of RaMBO: Blackbox differentiation for ranking

def ranker(arr: In[Array[float]], arr_size: In[int], reverse: In[int], rank: Out[Array[int]]):
    i: int = 0
    j: int = 0
    cnt: int = 0
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


def normalize_rank(rank: In[Array[int]], size: In[int], norm_rank: Out[Array[float]]):
    i: int = 0
    while (i < size, max_iter := 10000):
        norm_rank[i] = (rank[i] + 1.0) / (size + 1e-8)
        i = i + 1


def RaMBO_forward(score: In[Array[float, 1000]], size: In[int], rank: Out[Array[int]], norm_rank: Out[Array[float]]):
    ranker(score, size, 1, rank)
    normalize_rank(rank, size, norm_rank)


# recall loss according to https://github.com/martius-lab/blackbox-backprop/blob/master/blackbox_backprop/recall.py
def recall_loss(norm_rank: In[Array[float]], pos_norm_rank : Out[Array[float]], size: In[int], label: Out[Array[int]]) -> float:
    i: int = 0
    loss : float

    query_rank : Array[float, 20000]
    
    while (i < size, max_iter := 10000):
        query_rank[i] = (norm_rank[i] - pos_norm_rank[i]) * label[i]
        i = i + 1
    
    i = 0
    denominator: float

    while (i < size, max_iter := 10000):
        # denormalize ranks
        query_rank[i] = query_rank[i] * size
        loss = loss + query_rank[i] * label[i]
        denominator = denominator + label[i]
        i = i + 1

    loss = loss / denominator

    return loss


d_rev_recall_loss = rev_diff(recall_loss)


def rank_gradient(score: In[Array[float]], pos_score: In[Array[float]], size: In[int], rank: In[Array[int]], pos_rank: In[Array[int]], norm_rank: In[Array[float]],
                  pos_norm_rank: In[Array[float]], d_norm_rank: Out[Array[float]], label: In[Array[int]]):

    i: int = 0
    deviation: float

    # This is actually high enough as normalised ranks live in [0,1].
    HIGH_CONSTANT: float = 2.0
    # TINY_CONSTANT: float = 1e-5

    # score margin introduced in the paper
    while (i < size, max_iter := 10000):
        deviation = label[i] - 0.5
        # reuse pos_score to avoid modifying score
        score[i] = score[i] - 0.02 * deviation
        i = i + 1
    
    i = 0
    RaMBO_forward(score, size, rank, norm_rank)

    while (i < size, max_iter := 10000):
        pos_score[i] = -norm_rank[i] + HIGH_CONSTANT * label[i]
        i = i + 1
    
    RaMBO_forward(pos_score, size, pos_rank, pos_norm_rank)

    d_size: int
    dout : float = 1.0

    d_rev_recall_loss(norm_rank, d_norm_rank, pos_norm_rank, size,
                      d_size, label, dout)


# output gradient
def RaMBO_backward(score: In[Array[float]], d_score: Out[Array[float]], score_prime: In[Array[float]],
                   size: In[int], rank_prime: In[Array[int]], norm_rank: In[Array[float]], norm_rank_prime: In[Array[float]],
                   d_norm_rank: Out[Array[float]], lambda_val: In[float]):

    i: int

    while (i < size, max_iter := 10000):
        score_prime[i] = score[i] + lambda_val * d_norm_rank[i]
        i = i + 1

    RaMBO_forward(score_prime, size, rank_prime, norm_rank_prime)

    i = 0
    while (i < size, max_iter := 10000):
        d_score[i] = -(norm_rank[i] - norm_rank_prime[i]) / (lambda_val + 1e-8)
        i = i + 1


# wrap of the whole process
def call_RaMBO(score: In[Array[float]], d_score: Out[Array[float]], pos_score: In[Array[float]], size: In[int], rank: In[Array[int]], pos_rank: In[Array[int]],
               norm_rank: In[Array[float]], pos_norm_rank: In[Array[float]], d_norm_rank: Out[Array[float]], label: In[Array[int]], lambda_val: In[float]):
    rank_gradient(score, pos_score, size, rank, pos_rank,
                  norm_rank, pos_norm_rank, d_norm_rank, label)

    # restore norm_rank
    # RaMBO_forward(score, size, rank, norm_rank)

    # a liitle reuse of variables to ease function def
    RaMBO_backward(score, d_score, pos_score, size, pos_rank,
                   norm_rank, pos_norm_rank, d_norm_rank, lambda_val)
