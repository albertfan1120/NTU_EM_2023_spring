import numpy as np


def Pxy(m, n):
    return (100 - abs(m - n)) / 666700


def get_table(m_np, n_np):
    Pxy_table = np.zeros((m_np.shape[0], n_np.shape[0]))
    for i, m in enumerate(m_np):
        for j, n in enumerate(n_np):
            Pxy_table[i, j] = Pxy(m, n)
    return Pxy_table


def ce_532(m_np, n_np):
    Pxy_table = get_table(m_np, n_np)
    Px = Pxy_table.sum(axis = 0)
    Py = Pxy_table.sum(axis = 1)

    sum = 0
    for n in range(Pxy_table.shape[0]):
        sum -= Px[n] * np.log(Py[n])

    return sum


def ce_534(m_np, n_np):
    Pxy_table = get_table(m_np, n_np)
    Px = Pxy_table.sum(axis = 0)
    Py_x = np.diagonal(Pxy_table) / Px

    sum = 0
    for n in range(Pxy_table.shape[0]):
        sum -= Px[n] * np.log(Py_x[n])

    return sum


if __name__ == '__main__':
    m = np.arange(1, 101)
    n = np.arange(1, 101)
    ce = ce_532(m, n)
    print("Part a:", ce)

    ce = ce_534(m, n)
    print("Part b:", ce)