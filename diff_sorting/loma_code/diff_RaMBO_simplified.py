# Simplified version of RaMBO in loma which eased the array def / func def

def ranker(arr: In[Array[float]], arr_size: In[int], reverse: In[int], rank: Out[Array[int]]):
    i: int = 0
    j: int = 0
    cnt: int = 0
    while (i < arr_size, max_iter := 20000):
        cnt = 0
        j = 0
        while (j < arr_size, max_iter := 20000):
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
    while (i < size, max_iter := 20000):
        norm_rank[i] = (rank[i] + 1.0) / (size + 1e-8)
        i = i + 1


def RaMBO_forward(score: In[Array[float]], size: In[int], rank: Out[Array[int]], norm_rank: Out[Array[float]]):
    ranker(score, size, 1, rank)
    normalize_rank(rank, size, norm_rank)


# recall loss according to https://github.com/martius-lab/blackbox-backprop/blob/master/blackbox_backprop/recall.py
def recall_loss(norm_rank: In[Array[float]], pos_norm_rank: Out[Array[float]], size: Out[int], label: Out[Array[int]]) -> float:
    i: int = 0
    loss: float

    query_rank: Array[float, 20000]

    while (i < size, max_iter := 20000):
        query_rank[i] = (norm_rank[i] - pos_norm_rank[i]) * label[i]
        i = i + 1

    i = 0
    denominator: float

    while (i < size, max_iter := 20000):
        # denormalize ranks
        query_rank[i] = query_rank[i] * size
        loss = loss + query_rank[i] * label[i]
        denominator = denominator + label[i]
        i = i + 1

    loss = loss / denominator

    return loss


d_rev_recall_loss = rev_diff(recall_loss)


def rank_gradient(score: In[Array[float]], size: In[int], rank: In[Array[int]],
                  norm_rank: In[Array[float]], d_norm_rank: Out[Array[float]], label: In[Array[int]]):

    i: int = 0
    deviation: float

    # This is actually high enough as normalised ranks live in [0,1].
    HIGH_CONSTANT: float = 2.0
    # TINY_CONSTANT: float = 1e-5

    # score margin introduced in the paper
    while (i < size, max_iter := 20000):
        deviation = label[i] - 0.5
        # score margin proposed in paper
        score[i] = score[i] - 0.02 * deviation
        i = i + 1

    i = 0
    RaMBO_forward(score, size, rank, norm_rank)

    pos_score: Array[float, 20000]
    while (i < size, max_iter := 20000):
        pos_score[i] = -norm_rank[i] + HIGH_CONSTANT * label[i]
        i = i + 1

    pos_rank: Array[int, 20000]
    pos_norm_rank: Array[float, 20000]

    RaMBO_forward(pos_score, size, pos_rank, pos_norm_rank)

    dout: float = 1.0

    d_rev_recall_loss(norm_rank, d_norm_rank, pos_norm_rank, size, label, dout)


# output gradient
def RaMBO_backward(score: In[Array[float]], d_score: Out[Array[float]],
                   size: In[int], norm_rank: In[Array[float]], d_norm_rank: Out[Array[float]], lambda_val: In[float]):

    i: int

    score_prime : Array[float, 20000]

    while (i < size, max_iter := 20000):
        score_prime[i] = score[i] + lambda_val * d_norm_rank[i]
        i = i + 1

    rank_prime : Array[int, 20000]
    norm_rank_prime : Array[float, 20000]
    RaMBO_forward(score_prime, size, rank_prime, norm_rank_prime)

    i = 0
    while (i < size, max_iter := 20000):
        d_score[i] = -(norm_rank[i] - norm_rank_prime[i]) / (lambda_val + 1e-8)
        i = i + 1


# wrap of the whole process
def call_RaMBO(score: In[Array[float]], d_score: Out[Array[float]], size: In[int], rank: In[Array[int]], 
               norm_rank: In[Array[float]], d_norm_rank: Out[Array[float]], label: In[Array[int]], lambda_val: In[float]):
    rank_gradient(score, size, rank, norm_rank, d_norm_rank, label)

    # a liitle reuse of variables to ease function def
    RaMBO_backward(score, d_score, size, norm_rank, d_norm_rank, lambda_val)
