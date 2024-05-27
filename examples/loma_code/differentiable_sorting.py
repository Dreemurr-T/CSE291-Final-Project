# input an array
# output the sorted array

# RaMBO: Blackbox differentiation for ranking


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
    
