import numpy as np

def isotonic_l2(y, sol):
  """Solves an isotonic regression problem using PAV.

  Formally, it solves argmin_{v_1 >= ... >= v_n} 0.5 ||v - y||^2.

  Args:
    y: input to isotonic regression, a 1d-array.
    sol: where to write the solution, an array of the same size as y.
  """
  n = y.shape[0]
  target = np.arange(n)
  c = np.ones(n)
  sums = np.zeros(n)

  # target describes a list of blocks.  At any time, if [i..j] (inclusive) is
  # an active block, then target[i] := j and target[j] := i.

  for i in range(n):
    sol[i] = y[i]
    sums[i] = y[i]

  i = 0
  while i < n:
    k = target[i] + 1
    if k == n:
      break
    if sol[i] > sol[k]:
      i = k
      continue
    sum_y = sums[i]
    sum_c = c[i]
    while True:
      # We are within an increasing subsequence.
      prev_y = sol[k]
      sum_y += sums[k]
      sum_c += c[k]
      k = target[k] + 1
      if k == n or prev_y > sol[k]:
        # Non-singleton increasing subsequence is finished,
        # update first entry.
        sol[i] = sum_y / sum_c
        sums[i] = sum_y
        c[i] = sum_c
        target[i] = k - 1
        target[k - 1] = i
        if i > 0:
          # Backtrack if we can.  This makes the algorithm
          # single-pass and ensures O(n) complexity.
          i = target[i - 1]
        # Otherwise, restart from the same point.
        break

  # Reconstruct the solution.
  i = 0
  while i < n:
    k = target[i] + 1
    sol[i + 1 : k] = sol[i]
    i = k

y = np.array([64.2, 34.5, 25.5, 12.22, 11.0, 90.1])
sol = np.array([6, 5, 4, 3, 2, 1])

isotonic_l2(y, sol)
print(sol)