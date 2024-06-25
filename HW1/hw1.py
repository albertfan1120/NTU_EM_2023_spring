import numpy as np
import math
import matplotlib.pyplot as plt


def f_x_y(x, y):
    return 5 * math.cos(-abs(x*y) / 5)


def euler(x, init_val):
    y = np.zeros_like(x)
    y[0] = init_val

    N = x.shape[0]
    for i in range(1, N):
        y[i] = y[i-1] + f_x_y(x[i-1], y[i-1]) * (x[i] - x[i-1])
    
    return y


def mod_euler(x, init_val):
    y = np.zeros_like(x)
    y[0] = init_val

    N = x.shape[0]
    for i in range(1, N):
        h = x[i] - x[i-1]
        y_star = y[i-1] + h*f_x_y(x[i-1], y[i-1])
        y[i] = y[i-1] + (f_x_y(x[i-1], y[i-1]) + f_x_y(x[i], y_star)) * h / 2
    
    return y


def RK4(x, init_val):
    y = np.zeros_like(x)
    y[0] = init_val

    N = x.shape[0]
    for i in range(1, N):
        h = x[i] - x[i-1]
        k1 = f_x_y(x[i-1], y[i-1])
        k2 = f_x_y(x[i-1] + 0.5*h, y[i-1] + 0.5*h*k1)
        k3 = f_x_y(x[i-1] + 0.5*h, y[i-1] + 0.5*h*k2)
        k4 = f_x_y(x[i-1] + h, y[i-1] + h*k3)

        y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) * h / 6
    
    return y


def plot(x, y_eular, y_mod, y_RK4):
    plt.figure()
    plt.title('Eular method')
    plt.plot(x, y_eular)
    plt.show()

    plt.figure()
    plt.title('Modified Eular method')
    plt.plot(x, y_mod)
    plt.show()

    plt.figure()
    plt.title('RK4 method')
    plt.plot(x, y_RK4)
    plt.show()


if __name__ == '__main__':
    h = 0.01
    start, end = 0, 10
    N = int((end - start) / h + 1)
    
    init_val = 0
    x = np.linspace(start, end, num=N)
    
    y_eular = euler(x, init_val)
    y_mod = mod_euler(x, init_val)
    y_RK4 = RK4(x, init_val)
    
    plot(x, y_eular, y_mod, y_RK4)