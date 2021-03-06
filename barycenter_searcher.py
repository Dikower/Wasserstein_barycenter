# -*- coding: utf-8 -*-

"""
author - Din Dmitriy
"""

import os
import time
import tqdm
import imageio
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
from skimage.io import imsave, imread
from skimage.transform import resize
from skimage.color import rgb2gray


class WassersteinBarycenter:
    def __init__(self, input_path: str="images", image_size: int=10, gamma: float=0.9, eps: float=0.00000002,
                 visual: bool=False, subplots_per_line: int = 1, iter_number_between_visualisation: int=10):
        assert type(input_path) == str and type(image_size) == int and type(gamma) == type(eps) == float, \
            "Parameters must be of a certain type: path-str, image_size(side of a square)-int, gamma-float, eps-float"

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
        self._Q = [np.matrix(rgb2gray(resize(imread(self.input_path + i), (self._n_sqrt, self._n_sqrt)))).flatten()
                   for i in os.listdir(self.input_path)]
        self._m = len(self._Q)  # Amount of images
        self._debug_print("variable", f"_m = {self._m}")

        # visualisation part
        self._visual = visual
        self._subplots = None
        self._subplots_per_line = subplots_per_line
        self.iter_number_between_visualisation = iter_number_between_visualisation
        if self._visual:
            try:
                os.mkdir("history")
            except FileExistsError:
                pass

            plt.axis('off')
            plt.ion()
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.2, top=0.2, wspace=0.1, hspace=0.1)
            fig, subplots = plt.subplots(self._subplots_per_line, self._subplots_per_line, sharex=True, sharey=True)
            fig.patch.set_facecolor('black')
            if self._subplots_per_line == 1:
                subplots.set_facecolor('black')
                self._subplots = [subplots]
            else:
                self._subplots = []
                for group in subplots:
                    for subplot in group:
                        subplot.set_facecolor('black')
                        self._subplots.append(subplot)

        # variables for optimizer
        self._gamma = gamma
        self._L = 2 / self._gamma
        self._eps = eps
        self._lmbd = np.array([np.zeros(self._n) for _ in range(self._m)])
        self.p = self._Q
        self.save_results("start")
        # released optimization methods
        self._allowed_methods = {"nesterov_triangle": [self.Nesterov_Triangle_Method, (self._lmbd, self._eps)],
                                 "ibp": [self.Iterative_Bregman_Projections, ()]}

        # makes the sum of the q equal to 1
        self._Q = np.array([q * (1 / np.sum(q)) for q in self._Q])

        # makes the distance matrix
        self._C = self._distance_matrix()

    # universal method for debug
    def _debug_print(self, level: str, *values) -> None:
        if level in self.debug_variables:
            if self.debug_variables[level]:
                print(f"[{round(time.process_time() - self._summary_time, 2)}][{level.ljust(self.interval)}]"
                      f" - {', '.join(values)}")

    # makes distance matrix n*n
    def _distance_matrix(self) -> np.array:
        c = np.zeros((self._n, self._n))
        for i in range(self._n):
            for j in range(self._n):
               c[i, j] = (i - j) ** 2 + (j - i) ** 2
        return c

    # draws results of calculating
    def _visualise(self, p, k) -> None:
        if self._m > self._subplots_per_line ** 2:
            p = p[len(p) - self._subplots_per_line ** 2:] # not [:self.s_p_l ** 2] because doesnt work when m == s_p_l

        for i in range(len(p)):
            maximum = np.max(p[i])
            self._subplots[i].imshow(np.resize(p[i] / maximum, (self._n_sqrt, self._n_sqrt)), cmap="gray")
        plt.draw()
        plt.pause(0.0001)
        plt.savefig(f"history/{k}.jpg")

    # gradient of the sum
    def _summary_gradient(self, lmbd) -> np.array:
        self._debug_print("more_info", f" _summary_gradient: lmbd = {lmbd}")

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
        self._debug_print("more_info", f"result = {result}")

        # applies gradients to the original formula
        result = np.sum(result, axis=0)  # TODO clarify the formula
        self._debug_print("more_info", f"result shape = {result.shape}")
        self._debug_print("process", f"the gradient counted for {round(time.process_time() - start, 2)}s")
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
        self._debug_print("more_info", f"lmbd shape = {lmbd.shape}")
        self._debug_print("more_info", f"small gradient counted for {round(time.process_time() - start, 2)}s")
        return lmbd

    # new experimental method
    def Iterative_Bregman_Projections(self, args) -> None:
        # for one picture
        k = 0
        u = np.ones((self._m, self._n))
        v = np.ones((self._m, self._n))
        # print("C", self._C, "\nC/gamma", self._C/self._gamma)
        E = np.e ** (-self._C / self._gamma)
        np.savetxt("E.csv", E, delimiter=" ")
        # for e in E:
            # print(e, end=' ')
            # print('', end='\n')
        # print("+"*100)
        while True:
            k += 1
            # print(k)
            for i in range(self._m):
                v[i] = self._Q[i] / ((E.T).dot(u[i]))
                # print("Q[i]", self._Q[i])
                # print("v", v[i])
                # print("u*E.T", (E.T).dot(u[i]))

            p = u[0] * (E.dot(v[0]))
            for i in range(1, self._m):
                p *= u[i] * (E.dot(v[i]))
                # print("v*E.T", E.dot(v[i]))
                # print("p", p)
            if self._visual:
                if k % self.iter_number_between_visualisation == 0:
                    self._visualise([p], k)

            for i in range(self._m):
                u[i] = p / (E.dot(v[i]))
                # print("u", u[i])

            # print("_"*1000)
            if k >= 300:
                self.p = [p]
                self.save_results("result/")
                break

    # method of optimization
    def Nesterov_Triangle_Method(self, args) -> None:
        x, eps = args
        k = 0
        a = 0
        A = 0  # not constant
        u = np.copy(x)
        p = np.zeros((self._m, self._n))  # solutions of the task
        old_norm = 0
        start_norm = 0
        while True:
            k += 1
            start = time.process_time()
            self._debug_print("process", f"{k} iteration started")

            a = 1 / (2 * self._L) + (1 / (4 * self._L**2) + a**2) ** 0.5
            self._debug_print("more_info", f"a = {a}")

            last_A = A
            A += a
            self._debug_print("more_info", f"A = {A}")

            y = (a * u + last_A * x) / A
            self._debug_print("more_info", f"y = {y}")

            summary_gradient, small_gradients = self._summary_gradient(y)
            u = u - a * summary_gradient
            self._debug_print("more_info", f"u = {u}")

            x = (a * u + last_A * x) / A
            self._debug_print("more_info", f"x = {x}")

            p += a * small_gradients
            self.p = p

            # visualisation
            if self._visual:
                if k % self.iter_number_between_visualisation == 0:
                    self._visualise(p, k)

            norm = np.linalg.norm(self._summary_gradient(x))
            max_norm = np.max(norm)
            self._debug_print("variable", f"norm = {round(max_norm, 6)}")

            stop = max_norm < eps or max_norm == old_norm
            self._debug_print("process", f"{k} iteration ended in {round(time.process_time() - start, 2)}s")
            self._debug_print("process", "-" * 100)
            old_norm = max_norm
            if k == 1:
                start_norm = max_norm
            if stop or k >= 1000:
                self._debug_print("process", f"optimization finished with difference {start_norm - max_norm} "
                                             "between the starting and final normal")
                self.p = p
                break

    # method for saving results
    def save_results(self, path: str) -> None:
        assert self.p is not None, "You have to calculate barycenter to save it"
        # for correct saving
        if path[-1] != "/":
            path += "/"
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        for i in range(len(self.p)):  # if use self._m doesnt work correctly with ibp
            maximum = np.max(self.p[i])
            imsave(path + f"{i}.jpg", np.resize(self.p[i] / maximum, (self._n_sqrt, self._n_sqrt)))  # 255/max = x/gray

        if self._visual:
            self._debug_print("process", "Saving to the gif file")
            with imageio.get_writer('history.gif', mode='I', duration=0.2) as writer:
                for filename in tqdm.tqdm(sorted(os.listdir("history"), key=lambda x: int(x.split(".")[0]))):
                    image = imageio.imread(f"history/{filename}")
                    writer.append_data(image)

    # method for using optimizers
    def calculate(self, method: str) -> None:
        assert method in self._allowed_methods, \
            f"You can use only {', '.join(self._allowed_methods.keys())} method(s)"
        method, args = self._allowed_methods[method]
        method(args)

    # method for setting debug output
    def set_debug_parameters(self, variable: bool=True, process: bool=True, more_info: bool=False):
        self.debug_variables["variable"] = variable
        self.debug_variables["process"] = process
        self.debug_variables["more_info"] = more_info


if __name__ == '__main__':
    # Arguments for class: path, image_size
    wb = WassersteinBarycenter("images", 10, gamma=100., visual=False, subplots_per_line=2,
                               iter_number_between_visualisation=1)
    wb.calculate("ibp")
    # wb.calculate("nesterov_triangle")
    wb.save_results("result")
