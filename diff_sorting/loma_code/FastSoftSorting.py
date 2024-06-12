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


# input_s, input_w, do l2 regularization
def isotonic_l2(input_s: In[Array[float]], input_w: In[Array[float]], arr_size: In[int], solution: Out[Array[float]]):

    difference: Array[float, 20000]
    i : int = 0
    while (i < arr_size, max_iter := 20000):
        difference[i] = input_s[i] - input_w[i]
        i = i + 1
   
    target: Array[int, 20000]
    while (i < arr_size, max_iter := 20000):
        target[i] = i
        i = i + 1

    c: Array[float, 20000]
    while (i < arr_size, max_iter := 20000):
        c[i] = 1
        i = i + 1

    sums: Array[float, 20000]

    while (i < arr_size, max_iter := 20000):
        solution[i] = difference[i]
        sums[i] = difference[i]
        i = i + 1

    i = 0
    k : int
    sum_y : float
    sum_c : float
    prev_y : float

    while (i < arr_size, max_iter := 20000):
        k = target[i] + 1
        if k < arr_size:
            if solution[i] <= solution[k]:
                sum_y = sums[i]
                sum_c = c[i]

                prev_y = solution[k]
                sum_y = sum_y + sums[k]
                sum_c = sum_c + c[k]
                k = target[k] + 1
                # while (k == arr_size or prev_y > solution[k], max_iter := 20000):
                while (k == arr_size, max_iter := 20000):
                    prev_y = solution[k]
                    sum_y = sum_y + sums[k]
                    sum_c = sum_c + c[k]
                    k = target[k] + 1
                
                if k == arr_size:
                    solution[i] = sum_y / sum_c
                    sums[i] = sum_y
                    c[i] = sum_c
                    target[i] = k - 1
                    target[k - 1] = i
                    if i > 0:
                        i = target[i - 1]

                if prev_y > solution[k]:
                    solution[i] = sum_y / sum_c
                    sums[i] = sum_y
                    c[i] = sum_c
                    target[i] = k - 1
                    target[k - 1] = i
                    if i > 0:
                        i = target[i - 1]
            
        if solution[i] > solution[k]:
            i = k
        if k == arr_size:
            i = arr_size  # break
        
    
    i = 0
    j : int
    while (i < arr_size, max_iter := 20000):
        k = target[i] + 1
        j = i + 1
        while (j < k, max_iter := 20000):
            solution[j] = solution[i]
            j = j + 1
        i = k


def soft_sort(arr : In[Array[float]], arr_size : In[int], reverse: In[int], regularization_strength: In[float], sorted_arr: Out[Array[float]]):

    sign : int
    if reverse == 1:
        sign = 1
    else:
        sign = -1

    # initialize sorted_arr
    array: Array[float, 20000]

    i : int = 0
    while (i < arr_size, max_iter := 20000):
        array[i] = arr[i]
        i = i + 1

    
    input_w: Array[float, 20000]

    i = 0
    while (i < arr_size, max_iter := 20000):
        input_w[i] = (arr_size-i)/regularization_strength
        i = i + 1
    
    # multiply by sign
    i = 0
    while (i < arr_size, max_iter := 20000):
        array[i] = array[i] * sign
        i = i + 1
    
    permutation: Array[int, 20000]

    # argsort_bubble_sort(array, arr_size, 1, permutation)

    s: Array[float, 20000]
    while (i < arr_size, max_iter := 20000):
        s[i] = array[permutation[i]]
        i = i + 1
    
    solution: Array[float, 20000]
    isotonic_l2(s, input_w, arr_size, solution)
    
    i = 0
    # calculate result
    while (i < arr_size, max_iter := 20000):
        sorted_arr[i] = (input_w[i] - solution[i]) * sign
        i = i + 1


def soft_sort_1(arr : In[Array[float]], arr_size : In[int], reverse: In[int], regularization_strength: In[float], sorted_arr: Out[Array[float]]):

    sign : int
    if reverse == 1:
        sign = 1
    else:
        sign = -1

    # initialize sorted_arr
    array: Array[float, 20000]

    i : int = 0
    j : int = 0
    while (i < arr_size, max_iter := 20000):
        array[i] = arr[i]
        i = i + 1

    
    input_w: Array[float, 20000]

    i = 0
    while (i < arr_size, max_iter := 20000):
        input_w[i] = (arr_size-i)/regularization_strength
        i = i + 1
    
    # multiply by sign
    i = 0
    while (i < arr_size, max_iter := 20000):
        array[i] = array[i] * sign
        i = i + 1
    
    permutation: Array[int, 20000]

    i = 0
    j = 0
    tmp_index: int = 0
    swapped: int = 0

    while (i < arr_size, max_iter := 20000):
        permutation[i] = i
        i = i + 1
    i = 0
    while (i < arr_size, max_iter := 20000):
        j = 0
        swapped = 0
        while (j < arr_size - i - 1, max_iter := 20000):
            if reverse == 1:
                if arr[permutation[j]] < arr[permutation[j + 1]]:
                    tmp_index = permutation[j]
                    permutation[j] = permutation[j + 1]
                    permutation[j + 1] = tmp_index
                    swapped = 1
            else:
                if arr[permutation[j]] > arr[permutation[j + 1]]:
                    tmp_index = permutation[j]
                    permutation[j] = permutation[j + 1]
                    permutation[j + 1] = tmp_index
                    swapped = 1
            j = j + 1
        i = i + 1
        if swapped == 0:
            i = arr_size

    s: Array[float, 20000]
    while (i < arr_size, max_iter := 20000):
        s[i] = array[permutation[i]]
        i = i + 1
    
    solution: Array[float, 20000]
    difference: Array[float, 20000]
    i = 0
    while (i < arr_size, max_iter := 20000):
        difference[i] = s[i] - input_w[i]
        i = i + 1
   
    target: Array[int, 20000]
    while (i < arr_size, max_iter := 20000):
        target[i] = i
        i = i + 1

    c: Array[float, 20000]
    while (i < arr_size, max_iter := 20000):
        c[i] = 1
        i = i + 1

    sums: Array[float, 20000]

    while (i < arr_size, max_iter := 20000):
        solution[i] = difference[i]
        sums[i] = difference[i]
        i = i + 1

    i = 0
    k : int
    sum_y : float
    sum_c : float
    prev_y : float

    while (i < arr_size, max_iter := 20000):
        k = target[i] + 1
        if k < arr_size:
            if solution[i] <= solution[k]:
                sum_y = sums[i]
                sum_c = c[i]

                prev_y = solution[k]
                sum_y = sum_y + sums[k]
                sum_c = sum_c + c[k]
                k = target[k] + 1
                # while (k == arr_size or prev_y > solution[k], max_iter := 20000):
                while (k == arr_size, max_iter := 20000):
                    prev_y = solution[k]
                    sum_y = sum_y + sums[k]
                    sum_c = sum_c + c[k]
                    k = target[k] + 1
                
                if k == arr_size:
                    solution[i] = sum_y / sum_c
                    sums[i] = sum_y
                    c[i] = sum_c
                    target[i] = k - 1
                    target[k - 1] = i
                    if i > 0:
                        i = target[i - 1]

                if prev_y > solution[k]:
                    solution[i] = sum_y / sum_c
                    sums[i] = sum_y
                    c[i] = sum_c
                    target[i] = k - 1
                    target[k - 1] = i
                    if i > 0:
                        i = target[i - 1]
            
        if solution[i] > solution[k]:
            i = k
        if k == arr_size:
            i = arr_size  # break
        
    
    i = 0
    while (i < arr_size, max_iter := 20000):
        k = target[i] + 1
        j = i + 1
        while (j < k, max_iter := 20000):
            solution[j] = solution[i]
            j = j + 1
        i = k

    i = 0
    # calculate result
    while (i < arr_size, max_iter := 20000):
        sorted_arr[i] = (input_w[i] - solution[i]) * sign
        i = i + 1

# d_fast_soft_sorting = fwd_diff(soft_sort)
# d_fast_soft_sorting = fwd_diff(soft_sort_1)
d_fast_soft_sorting = rev_diff(soft_sort_1)

        
# d_fast_soft_sorting = rev_diff(soft_sort)