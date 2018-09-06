# -*- coding: utf-8 -*-

"""
author - Din Dmitriy
"""

import os
import time
import tqdm
import numpy as np
from multiprocessing.pool import ThreadPool
from skimage.io import imsave, imread
from skimage.transform import resize
from skimage.color import rgb2gray
from matplotlib import animation
from matplotlib import pyplot as plt

import warnings
# warnings.filterwarnings("ignore")

class WasserstainBarycenter:
    def __init__(self, input_path: str="images", image_size: int=10, gamma: float=1.9, eps: float=0.00000002, visual: bool=False):
        assert type(input_path) == str and type(image_size) == int and type(gamma) == type(eps) == float, \
            "parameters must be of a certain type: path-str, image_size(side of a square)-int, gamma-float, eps-float"

        # variables for debug
        self._summary_time = time.process_time()
        self.debug_variables = {"variable": True, "process": True, "more_info": False}
        self.interval = max(len(variable) for variable in self.debug_variables.keys()) - 1

        # variables for images
        self._n_sqrt = image_size
        self._n = self._n_sqrt ** 2  # the summary amount of pixels

        # for correct reading
        self.input_path = input_path + "/" if input_path[:-1] != "/" else input_path

        # reads and normalizes images
        self._Q = [rgb2gray(resize(imread(self.input_path + i), (self._n_sqrt, self._n_sqrt)))
                   for i in os.listdir(self.input_path)]
        self._m = len(self._Q)  # Amount of images
        self._debug_print("variable", "_m = {}".format(self._m))

        # variables for optimizer
        self._visual = visual
        self._gamma = gamma
        self._L = 2 / self._gamma
        self._eps = eps
        self._lmbd = np.array([np.zeros(self._n) for _ in range(self._m)])
        self.p = self._Q
        self.save_results("start")

        # released optimization methods
        self._allowed_methods = {"nesterov_triangle": [self.Nesterov_Triangle_Method, (self._lmbd, self._eps)],
                                 "iterative_bregman_projections": [self.Iterative_Bregman_Projections, ()]}

        # makes the sum of the q equal to 1
        self._Q = np.array([q * (1 / np.sum(q)) for q in self._Q])

        # makes the distance matrix
        self._C = self._distance_matrix()

    # universal method for debug
    def _debug_print(self, level: str, *values):
        if level in self.debug_variables:
            if self.debug_variables[level]:
                print("[{0}][{1}] - {2}".format(round(time.process_time() - self._summary_time, 2),
                                                level.ljust(self.interval, " "), " ".join(values)))

    # makes distance matrix n*n
    def _distance_matrix(self):
        c = np.zeros((self._n, self._n))
        for i in range(self._n):
            for j in range(self._n):
               c[i, j] = (i - j) ** 2 + (j - i) ** 2
        return c

    def _visualise(self, p):
        if self._m > 16:
            p = p[:16]
        plt.figure(figsize=(10, 6))
        plt.ion()
        for i, item in enumerate(p):
            plt.subplot()
            plt.axis('off')
            plt.imshow(np.reshape(item, (self._n_sqrt, self._n_sqrt)))
        plt.show()

    # gradient of the sum
    def _summary_gradient(self, lmbd) -> np.array:
        self._debug_print("more_info", " _summary_gradient: lmbd = {}".format(lmbd))

        # initializes threads
        pool = ThreadPool(processes=len(lmbd))
        self._debug_print("process", "starting the counting of gradient")
        start = time.process_time()
        async = [pool.apply_async(self._small_gradient, (self._Q[i], lmbd[i])) for i in range(self._m - 1)] + \
                [pool.apply_async(self._small_gradient, (self._Q[-1], -np.sum(lmbd[:-1], axis=0)))]

        # takes results of threading
        result = np.array([async[i].get() for i in range(self._m)])

        # copies for next usage
        gradients = np.copy(result)
        self._debug_print("more_info", "result = {}".format(result))

        # applies gradients to the original formula
        result = np.sum(result, axis=0)  # TODO clarify the formula
        self._debug_print("more_info", "{} - result shape".format(result.shape))
        self._debug_print("process", "the gradient counted for {}s".format(round(time.process_time() - start, 2)))
        return result, gradients

    # gradient of the term
    def _small_gradient(self, q, lmbd) -> np.array:
        start = time.process_time()
        self._debug_print("more_info", "small gradient counting started")

        # debug output part
        if self.debug_variables["more_info"]:
            rng = tqdm.tqdm(range(self._n))
        else:
            rng = range(self._n)

        # gradient calculation
        for u in rng:
            lmbd[u] = np.sum((q * np.e ** ((-self._C[u] + lmbd[u]) / self._gamma)) /
                             np.sum(np.e ** ((-self._C + lmbd) / self._gamma)))
        self._debug_print("more_info", "{} - lmbd shape".format(lmbd.shape))
        self._debug_print("more_info", "small gradient counted for {}s".format(round(time.process_time() - start, 2)))
        return lmbd

    def Iterative_Bregman_Projections(self, args):
        # for one picture
        k = 0
        u = np.ones((self._m, self._n))
        v = np.zeros((self._m, self._n))
        # print(self._C/self._gamma)
        E = np.e ** (-self._C / self._gamma)
        p = 1
        while True:
            k += 1
            for i in range(self._m):
                v[i] = self._Q[i] / u[i].dot(E.T)
            for i in range(self._m):
                p *= u[i] * (v[i].dot(E))
            for i in range(self._m):
                u[i] = p / (v[i].dot(E))
            if k >= 10:
                self.p = p
                self.save_results("result/")
                break

    # method of optimization
    def Nesterov_Triangle_Method(self, args) -> None:
        x, eps = args
        k = 0
        a = 0
        A = 0  # not constant
        u = np.copy(x)
        p = np.zeros((self._m, self._n))  # solutions of task
        old_norm = 0
        start_norm = 0
        while True:
            k += 1
            start = time.process_time()
            self._debug_print("process", "{} iteration started".format(k))

            a = 1 / (2 * self._L) + (1/(4 * self._L**2) + a**2) ** 0.5
            self._debug_print("more_info", "a = {}".format(a))

            last_A = A
            A += a
            self._debug_print("more_info", "A = {}".format(A))

            y = (a * u + last_A * x) / A
            self._debug_print("more_info", "y = {}".format(y))

            summary_gradient, small_gradients = self._summary_gradient(y)
            u = u - a * summary_gradient
            self._debug_print("more_info", "u = {}".format(u))

            x = (a * u + last_A * x) / A
            self._debug_print("more_info", "x = {}".format(x))

            p += a * small_gradients
            self.p = p
            self.save_results("backup/")
            # visualisation
            # if self._visual:
            #     self._visualise(p)

            norm = np.linalg.norm(self._summary_gradient(x))
            max_norm = np.max(norm)
            self._debug_print("variable", "norm = {}".format(round(max_norm, 6)))

            stop = max_norm < eps or max_norm == old_norm
            self._debug_print("process", "{0} iteration ended in {1}s".format(k, round(time.process_time() - start, 2)))
            self._debug_print("process", "-" * 100)
            old_norm = max_norm
            if k == 1:
                start_norm = max_norm
            if stop or k >= 1000:
                self._debug_print("process", "optimization finished with difference {} "
                                             "between the starting and final normal ".format(start_norm - max_norm))
                self.p = p
                break

    # method for saving results
    def save_results(self, path):
        assert self.p is not None, "You have to calculate barycenter to save it"
        # for correct saving
        if path[-1] != "/":
            path += "/"
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        for i in range(self._m):
            maximum = np.max(self.p[i])
            imsave(path + "{}.jpg".format(i), np.resize(self.p[i] / maximum, (self._n_sqrt, self._n_sqrt)))

    # method for using optimizers
    def calculate(self, method):
        assert method in self._allowed_methods, \
            "You can use only {} method(s)".format(", ".join(self._allowed_methods.keys()))
        method, args = self._allowed_methods[method]
        method(args)

    # method for setting debug output
    def set_debug_parameters(self, variable=True, process=True, more_info=False):
        self.debug_variables["variable"] = variable
        self.debug_variables["process"] = process
        self.debug_variables["more_info"] = more_info

        
if __name__ == '__main__':
    # Arguments for class: path, image_size
    wb = WasserstainBarycenter("images/", 10)
    # wb.calculate("iterative_bregman_projections")
    wb.calculate("nesterov_triangle")
    wb.save_results("result")
