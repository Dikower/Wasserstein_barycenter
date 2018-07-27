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
import matplotlib.pyplot as plt
from scipy.misc import imsave
import time


def F_grad(lmbd):
    pool = ThreadPool(processes=len(lmbd))
    async_results = [pool.apply_async(H_grad, (Q[l], lmbd[l])) for l in range(len(lmbd - 1))] + \
                    [pool.apply_async(H_grad, (Q[-1], lmbd[-1]))]
    result = np.array([async_results[i].get() - async_results[-1].get() for i in range(len(lmbd))])
    if debug:
        print("result is ready")
    return result


def H_grad(q, lmbd):
    for u in range(n):
        # lmbd[u] = sum([(q[j] * np.e ** ((-C[u, j] + lmbd[u]) / gamma)) / sum(
        #               [np.e ** ((-C[i, j] + lmbd[i]) / gamma) for i in range(n)]) for j in range(n)])
        lmbd[u] = np.sum((q * np.e ** ((-C[u] + lmbd[u]) / gamma)) / np.sum(np.e ** ((-C + lmbd) / gamma)))

    if debug:
        print("lmbd is ready")
    return lmbd


def nesterov_triangle_method(x, eps, task_gradient, L):
    k = 0
    a = 0
    A = 0
    u = copy.copy(x)
    sum_p = 0

    while True:
        k += 1
        if debug:
            print("-" * 100)
            print("New iteration")
        a = 1 / (2 * L) + math.sqrt(1/(4 * L**2) + a**2)
        if debug:
            print("a is ready")
        last_A = A
        A += a
        if debug:
            print("A is ready")
        y = (a * u + last_A * x) / A
        if debug:
            print("y is ready")
        grad = task_gradient(y)
        if debug:
            print("grad is ready")
        u = u - a * grad
        if debug:
            print("u is ready")
        x = (a * u + last_A * x) / A
        if debug:
            print("x is ready")
        sum_p += a * H_grad(Q[0], y[0])
        if debug:
            print("sum_p is ready")

        # plt.imshow(X=sum_p, cmap='gray')
        # plt.draw()

        norm = np.linalg.norm(task_gradient(x))
        print(norm)
        stop = norm < eps

        if stop:
            return sum_p/A


if __name__ == '__main__':
    debug = False
    start = time.process_time()
    n = 400  # Общее количество пикселей
    n_sqrt = int(math.sqrt(n))
    images = [Image.open("images/" + i, 'r').convert('L').resize((n_sqrt, n_sqrt)) for i in os.listdir("images")]
    m = len(images)  # Количество картинок
    imsave("low_pix.jpg", images[0])
    Q = np.array([np.array(pic.getdata(), dtype="float64") for pic in images])
    Q = np.array([q * (1/np.sum(q)) for q in Q])
    if debug:
        print("Q is ready")
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            C[i, j] = (i - j) ** 2 + (j - i) ** 2
    if debug:
        print("C is ready")
    gamma = 4
    L = 1/gamma
    eps = 0.003

    lmbd = np.array([[0 for _ in range(n)] for _ in range(m)])
    plt.axis([0, n_sqrt, 0, n_sqrt])

    # plt.ion()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #
    p = nesterov_triangle_method(lmbd, eps, F_grad, L).reshape((int(math.sqrt(n)), int(math.sqrt(n))))
    imsave('barycenter.jpg', p)
    # fig.savefig('barycenter.png')
    print(time.process_time() - start)
