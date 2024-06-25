import numpy as np
import math


def norm(p, w):
    return math.sqrt(np.sum(w * np.square(p)))


def inner_pruduct(p, phi, w):
    return np.sum(p * phi * w)


def gram_schmidt(p, w):
    out = np.zeros_like(p)
    out[0] = p[0] / norm(p[0], w)
    
    N = p.shape[0]
    for i in range(1, N):
        tmp = 0
        for j in range(i):
            tmp += inner_pruduct(p[i], out[j], w) * out[j]
        g = p[i] - tmp
        out[i] = g / norm(g, w)

    return out



if __name__ == '__main__':
    basis = np.zeros((5, 13))
    for k in range(5):
        for i in range(13):
            n = i-6
            basis[k, i] = pow(n, k)
    
    weight_a = np.ones((13))
    weight_b = np.zeros((13))
    for i in range(13): 
        n = i-6
        weight_b[i] = 1-abs(n)/7
    
    print("Part a:")
    orthornormal_set = gram_schmidt(basis, weight_a)
    print(orthornormal_set)

    print("Part b:")
    orthornormal_set = gram_schmidt(basis, weight_b)
    print(orthornormal_set)
