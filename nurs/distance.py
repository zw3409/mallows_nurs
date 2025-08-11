from __future__ import annotations
import numpy as np
import numba as nb


@nb.njit(cache=True, fastmath=True)
def l2(perm):
    s = 0
    for i in range(perm.size):
        d = int(perm[i]) - i
        s += d * d
    return s


@nb.njit(cache=True, fastmath=True)
def l1(perm):
    s = 0
    for i in range(perm.size):
        s += abs(int(perm[i]) - i)
    return s


@nb.njit(cache=True, fastmath=True)
def hamming(perm):
    s = 0
    for i in range(perm.size):
        s += perm[i] != i
    return s


@nb.njit(cache=True, fastmath=True)
def invert(perm):
    n = perm.size
    out = np.empty(n, np.int64)
    for i in range(n):
        out[perm[i]] = i
    return out


@nb.njit(cache=True, fastmath=True)
def lis_len(arr):
    n = arr.size
    tails = np.empty(n, np.int64)
    size = 0
    for x in arr:
        l = 0
        r = size
        while l < r:
            m = (l + r) >> 1
            if tails[m] < x:
                l = m + 1
            else:
                r = m
        tails[l] = x
        if l == size:
            size += 1
    return size


@nb.njit(cache=True, fastmath=True)
def lds_len(arr):
    n = arr.size
    neg = np.empty(n, np.int64)
    for i in range(n):
        neg[i] = -arr[i]
    return lis_len(neg)


@nb.njit(cache=True, fastmath=True)
def ulam(perm):
    return perm.size - lis_len(invert(perm))


@nb.njit(cache=True, fastmath=True)
def inv(perm):
    n = perm.size
    bit = np.zeros(n + 1, np.int64)
    inv_count = 0
    seen = 0
    for v in perm:
        x = v + 1
        s = 0
        i = x
        while i:
            s += bit[i]
            i &= i - 1
        inv_count += seen - s
        seen += 1
        i = x
        while i <= n:
            bit[i] += 1
            i += i & -i
    return inv_count


@nb.njit(cache=True, fastmath=True)
def dist(dist_id, perm):
    if dist_id == 0:
        return l2(perm)
    elif dist_id == 1:
        return l1(perm)
    elif dist_id == 2:
        return hamming(perm)
    elif dist_id == 3:
        return ulam(perm)
    else:
        return inv(perm)


DIST_NAME_TO_ID = {
    "L2": 0,
    "L1": 1,
    "Hamming": 2,
    "Ulam": 3,
    "inv": 4
}