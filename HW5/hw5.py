import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    M = 20
    m = np.arange(M) + 1
    sin = np.sin(m*np.pi/(4*M))
    X = np.vstack((m, sin)).T
    
    X_mean = X.mean(axis = 0)
    A = X - X_mean
    U, S, Vh = np.linalg.svd(A)
    V1 = Vh[0]
    
    start = -(M//2+1)
    end = (M//2+1)
    N = int((end - start) / 0.01)
    reg_c = np.linspace(start, end, N, endpoint=True).reshape(-1,1)

    line = X_mean + reg_c.dot( V1.reshape(1,-1) )
    
    plt.figure()
    plt.scatter(m, sin, c = 'g',label='original point')
    plt.plot(line[:, 0], line[:, 1], label='regression line')
    plt.xlabel('[x]1')
    plt.ylabel('[x]2')
    plt.legend()
    plt.show()