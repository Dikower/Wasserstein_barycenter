# -*- coding: utf-8 -*-

"""
Place square images into dir 'images' result will be saved as 'outfile.jpg'

was coded 23.07.2018 at night by Dikower
"""

import numpy as np
import math
import os
import copy
from PIL import Image
from multiprocessing.pool import ThreadPool
from scipy.misc import imsave


def F_grad(lmbd):
    pool = ThreadPool(processes=len(lmbd))
    async_results_H_grad_minus = [pool.apply_async(H_grad, (Q[m-1], -sum(lmbd[:m-1]))) for l in range(len(lmbd))]
    async_results = [pool.apply_async(H_grad,(Q[l], lmbd[l]))  for l in range(len(lmbd))]
    return np.array([async_results[i].get() - async_results_H_grad_minus[i].get() for i in range(len(lmbd))])


def H_grad(q, lmbd):
    for u in range(n):
        lmbd[u] = sum([(q[j] * math.e ** ((-C[u][j] + lmbd[u]) / gamma)) / sum(
            [math.e ** ((-C[i][j] + lmbd[i]) / gamma) for i in range(n)]) for j in range(n)])
    # print(lmbd)
    return lmbd


def nesterov_triangle_method(x, eps, task_gradient, L):
    k = 0
    a = 0
    A = 0
    u = copy.copy(x)
    sum_p = 0

    while True:
        k += 1

        a = 1 / (2 * L) + math.sqrt(1/(4 * L**2) + a**2)
        last_A = A
        A += a

        y = (a * u + last_A * x) / A
        grad = task_gradient(y)
        u = u - a * grad
        x = (a * u + last_A * x) / A
        sum_p += a * H_grad(Q[0], y[0])

        norm = np.linalg.norm(task_gradient(x))
        print(norm)
        stop = norm < eps
        if stop:
            return sum_p/A


if __name__ == '__main__':

    n = 100  # Общее количество пикселей
    n_sqrt = int(math.sqrt(n))
    images = [Image.open("images/" + i, 'r').convert('L').resize((n_sqrt, n_sqrt)) for i in os.listdir("images")]
    m = len(images)  # Количество картинок

    Q = np.array([np.array(pic.getdata(), dtype="float64") for pic in images])
    Q = np.array([q * (1/np.sum(q)) for q in Q])

    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            C[i, j] = (i - j) ** 2 + (j - i) ** 2

    gamma = 4  # Makes gradient descent faster
    L = 1/gamma
    eps = 0.3

    lmbd = np.array([[0 for _ in range(n)] for _ in range(m)])
    p = nesterov_triangle_method(lmbd, eps, F_grad, L).reshape((int(math.sqrt(n)), int(math.sqrt(n))))
    imsave('outfile.jpg', p)
