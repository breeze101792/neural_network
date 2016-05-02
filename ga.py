#!/bin/python3
import numpy as np
import math
import random


class ga:

    def __init__(self, dataset, eval_func, population=10, crossover_rate=0.6, mutation_rate=0.2):
        self.population = population
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.eval_func = eval_func

        self.dataset = dataset
        self.pool = []
        self.co_pool = [[]]
        # self.population = len(dataset)
        self.data_dimension = len(dataset[0])
        self.mutation_scal = 0.1

        self.vec_size = len(dataset[0])
        self.best_vec = None

    def find_best(self):
        best_man = 0
        f_of_x = np.ones(len(self.pool))
        for idx, each_x in enumerate(self.pool):
            # print("start of loop")
            # print(each_x)
            # print(idx)
            # print(f_of_x[idx])
            # print(eval_func(each_x))
            f_of_x[idx] = self.eval_func(each_x)
            if f_of_x[best_man] < f_of_x[idx]:
                best_man = idx
        # print("best\t-> ", self.pool[best_man], f_of_x[idx])
        return [best_man, f_of_x[best_man]]

    def pool_init(self):
        # TODO ORI
        # for _ in range(self.population):
        #     self.pool.append(
        #         self.dataset[random.randint(0, len(self.dataset) - 1)])

        # TODO random
        # print(self.dataset)
        # print(self.vec_size)
        for _ in range(self.population):
            self.pool.append(np.random.rand(self.vec_size) * 2 - 1)
        self.best_vec = [0, self.pool[0]]
        # print("pool init\t->", self.pool)

    def cp_to_pool(self):
        f_of_x = np.zeros(self.population)
        # print("start of cp to pool~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print("pool\t", self.pool)
        for idx, each_x in enumerate(self.pool):
            # print(each_x, ",", self.eval_func(each_x))
            f_of_x[idx] = self.eval_func(each_x)

        max_idx = 0
        self.co_pool = []
        # print("test\t", f_of_x)
        f_mean = f_of_x.sum() / self.population
        # TODO for best ans
        for idx in range(self.population - 1):
            if f_of_x[idx] > f_of_x[max_idx]:
                max_idx = idx
            x_nan_check = f_of_x[idx] / f_mean
            if math.isnan(x_nan_check):
                continue
            for _ in range(int(round(x_nan_check))):
                if self.population - len(self.co_pool) == 0:
                    # print("skip ", self.population - len(self.co_pool))
                    break
                self.co_pool.append(self.pool[idx])
                # print("cp_to_pool -> ", self.pool[idx])
        # print("pool\t", self.pool)
        # print("f of x, mean\t", f_of_x, ", ", f_of_x / f_mean, ", ", f_mean)
        # print("co_pool\t", self.co_pool)

        # print(len(self.co_pool), ", ", self.population)
        for _ in range(self.population - len(self.co_pool)):
            self.co_pool.append(self.pool[max_idx])
        if f_of_x[max_idx] > self.best_vec[0]:
            self.best_vec[1] = self.pool[max_idx]
            self.best_vec[0] = f_of_x[max_idx]
        # print("co pool\t->",self.co_pool)
        # print("end of copy")
        # input()
    def cross_over(self):
        # print("start cross over~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``")
        for idx in range(len(self.co_pool)):
            if random.random() < self.crossover_rate:
                # print("idx\t", idx)
                if random.random() > 0.5:
                    # print("+")
                    pho = random.random()
                    x2_idx = random.randint(0, self.population - 1)

                    tmp = self.co_pool[idx] - self.co_pool[x2_idx]
                    self.co_pool[idx] = self.co_pool[idx] + pho * tmp
                    self.co_pool[x2_idx] = self.co_pool[x2_idx] - pho * tmp
                    # print("tmp ->", tmp, ", ", pho)
                    # print(x2_idx, ", ", self.co_pool)
                else:
                    # print("-")
                    pho = random.random()
                    x2_idx = random.randint(0, self.population - 1)

                    tmp = self.co_pool[x2_idx] - self.co_pool[idx]
                    self.co_pool[idx] = self.co_pool[idx] + pho * tmp
                    self.co_pool[x2_idx] = self.co_pool[x2_idx] - pho * tmp
                    # print("tmp ->", tmp, ", ", pho)
                    # print(x2_idx, ", ", self.co_pool)
        self.pool = self.co_pool
        # print("cross over\t-> ", self.co_pool)
        # input()

    def mutation(self):
        for idx in range(self.population):
            if random.random() > self.mutation_rate:
                # print(self.pool[idx])
                # print(self.pool)
                # np.random.rand(self.data_dimension) * self.mutation_scal
                self.pool[idx] = self.pool[idx] + np.random.rand(self.data_dimension) * 2 - 1
                # print(np.random.rand(self.data_dimension))
                # print(self.pool[idx])
                # print(self.pool)
                # print("mutate")
        # print("mutation\t-> ", self.pool)

    def time_flow(self, iteration):
        # print("ga data", self.dataset)
        for idx in range(iteration):

            self.cp_to_pool()
            self.cross_over()
            self.mutation()
            # print(self.pool)
            tmp = self.find_best()
            print("find best\t", idx, ", ", tmp)
            # if tmp[1] > 1.5:
            #     return self.pool[tmp[0]]
            # print("\n\n\n")
        print("last best\t", self.best_vec[0])
        # return self.pool[tmp[0]]
        # print(len(self.pool) ", ", len(self.co_pool))


class rbfn:

    def __init__(self, dataset, size=15, population=10, crossover_rate=0.6, mutation_rate=0.2):
        self.population = population
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        # first layer size
        self.size = size
        self.theta = 0.1
        self.weights = None
        self.mean = None
        self.sd = None
        self.data_size = len(dataset[0][0])
        self.dataset = dataset
        self.weight_init()

        self.ga = ga([np.concatenate([self.mean, self.sd, self.weights, [self.theta]])],
                  self.eval_func, population=self.population)
        self.ga.pool_init()

    def weight_init(self):
        # self.weights = np.ones(self.size)
        # self.mean = np.ones(self.size * self.data_size)
        # self.sd = np.ones(self.size)

        self.weights = np.random.rand(self.size)
        self.mean = np.random.rand(self.size * self.data_size)
        self.sd = np.random.rand(self.size) * 5
        # print(self.weights)
        # print(self.mean)
        # print(self.sd)
    # @staticmethod

    def eval_func(self, vec):
        tmp_list = np.zeros(self.size)
        # print("vec\t->", vec)
        # print(len(vec))
        # self.set_test()
        # vec = np.concatenate([self.mean, self.sd, self.weights, [self.theta]])
        engery = 0
        mean = vec[0: len(self.mean)]
        sd = vec[len(mean): len(mean) + len(self.sd)]
        weights = vec[len(mean) + len(self.sd):-1]
        theta = vec[-1]  # theta
        # print("weight\t", weights)
        # print("sd\t", sd)
        # print("mean\t", mean)
        # print("theta\t", theta)

        for idx, data in enumerate(self.dataset[0]):
            for idx_cell in range(self.size):
                tmp_list[idx_cell] = np.exp(np.power(data - mean[idx_cell * self.data_size:(
                    idx_cell + 1) * self.data_size], 2).sum() / (-2 * sd[idx_cell] * sd[idx_cell])) * weights[idx_cell]
            # print("data, idx, ", data, ", " , 'idx', ", " , tmp_list.sum() + theta)

            self.mean = vec[0: len(self.mean)]
            self.sd = vec[len(self.mean): len(self.mean) + len(self.sd)]
            self.weights = vec[len(self.mean) + len(self.sd):-1]
            self.theta = vec[-1]
            # print("testing eval\t", self.testing(data))
            # print("eval", data,  ", ",  self.dataset[1][idx], ", ", tmp_list.sum() + theta, ", ", self.dataset[1][idx] - (tmp_list.sum() + theta))
            engery += np.power(self.dataset[1]
                               [idx] - (tmp_list.sum() + theta), 2)
        # print("En\t", engery/2, ", ",  2 / (engery + 0.00001))
        return 2 / (engery + 0.00001)

    def traning(self, iteration = 10):
        # print("start traning")
        # self.eval_func(np.concatenate([self.mean, self.sd, self.weights, [self.theta]]))
        # print(self.iteration, self.population)
        self.ga.time_flow(iteration)
        vec = self.ga.best_vec[1]

        print("vec last\t", vec)

        self.mean = vec[0: len(self.mean)]
        self.sd = vec[len(self.mean): len(self.mean) + len(self.sd)]
        self.weights = vec[len(self.mean) + len(self.sd):-1]
        self.theta = vec[-1]
        return self.ga.best_vec[0]

    def set_test(self):
        self.theta = 0.797393474643501
        self.weights = np.array([-34.5605815967538, -11.4888723133404, 40.1648226722614,
                                 -23.2444312025007, -37.9146510385307, 30.181596364083,
                                 12.8166437526874, -42.2795622690169, -26.0953017174212,
                                 16.7380268841755, 35.1333171048022, -16.015759023398,
                                 -1.4983340292115, -51.5614546851866, 31.7235307600459,
                                 ])
        self.mean = np.array([43.1819122944494, 4.67694912179188, 41.3305084623705,
                              1.17018941008931, 6.95914660044372, 14.9646728049397,
                              10.1933323936358, 36.088159897427, 5.71624425736373,
                              39.6026107099369, 20.7589596939429, 33.6717831556713,
                              33.0252120207558, 1.58930899928984, 22.545622283574,
                              8.91770661306126, 30.1545284084815, 15.8006391642576,
                              36.3285621907462, 34.3852219789756, 29.4412741047615,
                              12.145057667796, 20.7664974181545, 42.6613474794298,
                              29.0429926483756, 35.8496023431405, 4.03250011816113,
                              30.0408060714296, 2.06146889645933, 11.1399384029959,
                              23.8122654091866, 24.7191155229264, 8.98598575546845,
                              36.0420209201945, 41.0903806012175, 17.7378095065447,
                              20.7687732646214, 6.45132502685707, 14.4731639280402,
                              4.20976062818233, 40.6158145760683, 23.4399616715872,
                              30.0558799787797, 14.3724289062527, 27.5806442725034,
                              ])
        self.sd = np.array([14.4507986690793, 12.5062079639388, 0.7785830957332,
                            10.505080463063, 5.3985122199679, 0.506918985286548,
                            14.4346393277141, 9.59576065767488, 3.13280453625204,
                            0.506147684573676, 9.39827467972426, 9.54200990754601,
                            14.4609910091532, 5.09517645454874, 4.66111009843079])

    def testing(self, data):

        if len(data) != self.data_size:
            print("data mismatch!")
            return -1
        tmp_list = np.zeros(self.size)
        # print("data\t=", data)
        # print("theta\t=", self.theta)
        # print("mean\t=", list(self.mean))
        # print("sd\t=", list(self.sd))
        # print("weights\t=", list(self.weights))
        for idx_cell in range(self.size):
            # print("data-dif\t", data - self.mean[idx_cell * self.data_size: (
            #     idx_cell + 1) * self.data_size])
            # print("data-dif pow\t", np.power(data - self.mean[idx_cell * self.data_size: (
            #     idx_cell + 1) * self.data_size], 2))
            # print("did sd\t",np.power(data - self.mean[idx_cell * self.data_size: (
            #     idx_cell + 1) * self.data_size], 2).sum() / (-2 * self.sd[idx_cell] * self.sd[idx_cell]))
            # print("weight\t", np.exp(np.power(data - self.mean[idx_cell * self.data_size: (
            # idx_cell + 1) * self.data_size], 2).sum() / (-2 *
            # self.sd[idx_cell] * self.sd[idx_cell])) * self.weights[idx_cell])
            # print(self.mean[idx_cell * self.data_size])
            tmp_list[idx_cell] = np.exp(np.power(data - self.mean[idx_cell * self.data_size: (
                idx_cell + 1) * self.data_size], 2).sum() / (-2 * self.sd[idx_cell] * self.sd[idx_cell])) * self.weights[idx_cell]
            # print("idx - " , idx_cell)
            # print("mean", self.mean[idx_cell * self.data_size: (idx_cell + 1) * self.data_size])
            # print("idx:idx to \t->", idx_cell * self.data_size,", ",  (idx_cell + 1) * self.data_size)
            # print("tmp_list\t->", tmp_list)
        # print("result\t=", list(tmp_list))
        return tmp_list.sum() + self.theta


def eval_func(val):
    tmp = -np.power(val[0] - 6, 2) + -np.power(val[1] - 8, 2) + 40
    print("eval", tmp if tmp > 0 else 0)
    return tmp if tmp > 0 else 0

if __name__ == '__main__':
    #
    # test = ga([np.array([1.0, 1]), np.array([6.0, 8]), np.array([10.0, 8]), ], eval_func, mutation_rate=0.1,
    #           crossover_rate=0.5, iteration=20, population=100)  # np.array([5.0,3]), np.array([19.0,24]
    # print(test.time_flow())

    # rb = rbfn(([np.array([0.]), np.array([1.])], [0.4, 0.1]),
    #           3, iteration=30, population=40)
    # rb.traning()
    # print("test data")
    # print(rb.testing(np.array([0.])))
    # print(rb.testing(np.array([0.5])))
    # print(rb.testing(np.array([1])))

    rb = rbfn(([np.array([15.3978641292678, 30.0, 11.532046954189472]), np.array([1.,2,2])], [0.3, 0.8]),
              15, iteration=1, population=1)
    # rb.traning()
    rb.set_test()
    print("GA->", rb.testing([15.3978641292678,30.0,11.532046954189472]))
