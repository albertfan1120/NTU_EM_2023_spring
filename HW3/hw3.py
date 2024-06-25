import numpy as np
import math
import matplotlib.pyplot as plt


def sample_x(start, end, delta):
    N = int((end - start) / delta + 1)
    return np.linspace(start, end, num=N)


def sample_f(delta, N):
    delta_f = 1 / (delta * N)
    fs = 1 / delta
    m = np.arange(N)
    f = delta_f * m
    f = np.concatenate((f[N//2+1:] - fs, f[:N//2+1]))
    return f


def DFT(g):
    G = np.zeros_like(g, dtype = 'complex_')
    N = g.shape[0]
    for m in range(N):
        sum = 0
        for n in range(N):
            sum += g[n] * np.exp(-1j * 2 * math.pi * m * n / N)
        G[m] = sum 
    shift_G = np.concatenate((G[N//2+1:],G[:N//2+1]))
    return shift_G



def modulation(G, f, x0):
    exp =  np.exp(-1j * 2 * math.pi * x0 * f)
    return exp * G
    

def plot(G, x):
    real, imag = np.real(G), np.imag(G)
    plt.figure()
    plt.plot(x,real,label='real part')
    plt.plot(x,imag,label='imag part')
    plt.xlabel('f')
    plt.ylabel('G(f)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    print('Part a:')
    delta = 0.05
    x = sample_x(-4, 4, delta)
    g = np.zeros_like(x)
    for i in range(x.shape[0]):
        g[i] = math.exp(-math.sqrt(abs(x[i]))) - math.exp(-2)
    G = DFT(g) * delta
    f = sample_f(delta, G.shape[0])
    G = modulation(G, f, -4)
    print(G)
    plot(G, f)

    print('\n\n')
    print('Part b:')
    delta = 0.1
    x = sample_x(0, 3, delta)
    g = np.zeros_like(x)
    for i in range(x.shape[0]):
      g[i] = math.sin(math.pi * (x[i] ** 2) / 9)
    G = DFT(g) * delta
    f = sample_f(delta, G.shape[0])
    G = modulation(G, f, 0)
    print(G)
    plot(G, f)
   