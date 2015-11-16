#!/usr/bin/env python3
#import matplotlib.pyplot as plt
import numpy as np
import math
import random
from data_proc import data_proc as dp

class cell:
    def __init__(self, dimensions = 2, learning_rate = 0.4, lr_coefficent = 1):
        self.dimensions = dimensions
        self.learning_rate = learning_rate
        self.xs = 0
        self.y = 0
        self.weights = None
        self.in_cells = []
        self.out_cells = []
        self.lr_coefficent = lr_coefficent
        self.dirrection = 0
        self.weight_init()
        # print("dim -> ", dimensions)
    def weight_init(self):
        # self.weights = [-1]
        # for i in range(0, self.dimensions):
        #     self.weights.append(0 if i % 2 == 0 else 1)

        tmp = [-1]
        for i in range(0, self.dimensions):
            tmp.append(0 if i % 2 == 0 else 1)
        self.weights = np.array(tmp)
        print(self.weights)
    def summation(self):
        x = 0
        x = np.sum(self.xs * self.weights)
        # for i in range(0, self.dimensions + 1):
        #     x += self.weights[i] * self.xs[i]
        #print("sum")
        #print(self.weights,", ", self.xs,", ", x)
        return x
    def activation(self):
        stimulates = self.summation()
        #print("stimulation = ", stimulates)
        if stimulates >= 0:
            #print("y >= 0")
            self.y = 1
            return self.y
        elif stimulates < 0:
            #print("y < 0")
            self.y = -1
            return self.y
    def feedback(self):
        # print("before ", self.weights , ", ", self.xs)
        # for i in range(0, len(self.weights)):
        #     if self.y < 0:
        #         self.weights[i] += self.xs[i] * self.learning_rate
        #         #print("after < ", self.weights , ", ", self.xs)
        #     elif self.y >= 0:
        #         self.weights[i] -= self.xs[i] * self.learning_rate
        #         print("after >= ", self.weights , ", ", self.xs)
        if self.y < 0:
            self.weights = self.weights + self.xs * self.learning_rate
                #print("after < ", self.weights , ", ", self.xs)
        elif self.y >= 0:
            self.weights = self.weights - self.xs * self.learning_rate

    def stimulation(self, stimulates):

        # self.xs = [-1]
        # self.xs.extend(stimulates)
        #print("start", self.weights, self.xs)
        tmp = [-1]
        tmp.extend(stimulates)
        self.xs = np.array(tmp)
        return self.activation()

    def getweights(self):
        return self.weights
    def get_weights(self):
        return self.weights
    def get_last_y(self):
        return self.y
    def __str__(self):
        print("test")

class mlp:
    def __init__(self, structure = [2,1], class_middle = [0.25, 0.75], err_rate = 0.9,  mse_max = 0.001, dimension = 2, learning_rate = 0.5, numofclass = 2, training_times = 10, best_w = False):
        self.cells = []
        self.learning_rate = learning_rate
        self.training_times = training_times
        self.dimension = dimension
        self.err_rate = err_rate
        # self.judge = judge
        self.itimes = 0
        self.best_w = best_w
        self.best_d = [0,0]

        # structure is [num of lay1, num of lay2, ...., num of layp]
        self.hit_rate = 1 - err_rate
        self.structure = structure
        self.class_middle = class_middle
        self.mse_max = mse_max
        self.numofhiddenlayer = len(structure) - 1
        self.numofoutput = 1
        self.numofclass = len(class_middle)
        self.struct_init()
    def struct_init(self):
        self.cells = []
        tmp_cell = []
        num_of_inputs = self.dimension

        for lay in range(len(self.structure)):
            tmp_cell = []
            for cell in range(self.structure[lay]):
                tmp_cell.append(cell_sigmoid(num_of_inputs, self.learning_rate))
            num_of_inputs = self.structure[lay]
            self.cells.append(tmp_cell)
        # tmp_cell = []
        # for cell in range(self.numofoutput):
        #     tmp_cell.append(cell_sigmoid(num_of_inputs, self.learning_rate))
        # self.cells.append(tmp_cell)
        #TODO multilayer
        print(self.cells)
    def classifier(self, data_set):
        result_ys = []

        min_c = 0
        # TODO nedd to be chaned!!!!!!
        for point in data_set:
            out = self.__forward(point)[0]
            # TODO mutilp
            for c in self.class_middle:
                if abs(out - min_c) > abs(out - c):
                    min_c = c
            result_ys.append(min_c)

            # tmp_point.append([self.cells[0][0].get_last_y(), self.cells[0][1].get_last_y()])
            # tmp_o.append(out)

        return result_ys

    def training(self, training_set):
        print(len(training_set[0]))
        points_set = training_set[0]
        expected_output_set = training_set[1]

        self.itimes = 0

        tmp_point = []
        tmp_y = []
        mse = 1
        tmp_mse = 0
        feedback_flag = False
        err_flag = False
        for it in range(self.training_times):
            tmp_point = []
            tmp_o = []
            mse = 0
            for point, ds in zip(points_set, expected_output_set):
                # self.__backward([y], self.__forward(point))
                # print("point, d ->\t", point,", ",ds)
                #TODO DATA trainsform
                ds = [ds]
                os = self.__forward(point)
                # print("os-before ->\t", os)

                #mse
                # tmp_mse = 0
                # for idx in range(len(self.cells[-1])):
                #     tmp_mse += (ys[idx] - os[idx]) * (ys[idx] - os[idx])
                # mse += tmp_mse


                feedback_flag = True
                for idx in range(len(self.cells[-1])):
                    min_c = self.class_middle[0]
                    for c in self.class_middle:
                        if abs(os[idx] - min_c) > abs(os[idx] - c):
                            min_c = c
                    # print("class_middle\t", min_c)
                    # print("min_c, ys->idx", min_c, ds[idx])
                    # print("back or not\t", min_c != ds[idx])
                    if min_c == ds[idx]:
                        feedback_flag = False
                            # print("feedback false")

                if feedback_flag:
                    self.__backward(ds, os)

                # elif err_flag:
                #     print("point , y", point, ", ",  ds)
                #     print("lastest_weight\t", self.cells[-1][0].get_weights())
                #     print("hid1_weight\t", self.cells[-2][0].get_last_y)
                #     print("hid2_weight\t", self.cells[-2][1].get_last_y)

                    # tmp = self.get_hit_rate(training_set)[0]
                    # if tmp == 0:
                    #     print("err, it, idx", tmp, it, idx)
                    #     input("stop")

                # print("os-after ->\t", self.__forward(point))#, "\n")
                # print("end!\n")
                tmp_point.append([self.cells[0][0].get_last_y(), self.cells[0][1].get_last_y()])
                tmp_o.append(ds)
            mes = mse / len(training_set[0])
            suc, err  = self.testing(training_set)
            # print("it, err\t", it, len(err[0])/(len(err[0]) + len(suc[0])) * 100, ", " , self.err_rate)
            if len(err[0])/(len(err[0]) + len(suc[0])) * 100 <= self.err_rate:
                # err_flag = True
                self.itimes = it + 1
                return self.itimes, suc, err
            # print("mse\t", mse)
            # if self.mse_max > mse:
            #     return tmp_point, tmp_y
        self.itimes = it + 1
        return self.itimes ,suc, err

    def test_train(self, training_set):
        self.cells[0][0].set_weights([-1.2,1,1])
        self.cells[0][1].set_weights([0.3,1,1])
        self.cells[1][0].set_weights([0.5,0.4,0.8])
        p = training_set[0]
        a = training_set[1]
        for idx in range(5000):

            for x,y in zip(p,a):
                out = self.__forward(x)
                # print("\nbefore", out)
                self.__backward([y], self.__forward(x))
                aout = self.__forward(x)
                # if(out == aout):
                #     print(idx)
                #     input("equ")
                # print("after ad\t", aout)

            print("\n")
    def __forward(self, point):
        tmp = []
        yi = point
        for lay in range(self.numofhiddenlayer + 1):
            tmp = []
            for cell in self.cells[lay]:
                tmp.append(cell.stimulation(yi))
            yi = tmp
        return yi
    # ys is like [3.9, 5.8], the number is depend on the number of output
    def __backward(self, ds, os):
        # print("backward")
        delta = []
        # print("ds, os\t", ds,", ", os)

        #output delta
        for idx, cell_j in enumerate(self.cells[-1]):
            delta.append([(ds[idx] - os[idx]) * os[idx] * (1 - os[idx])])
        # print("delta\t", delta)

        #hidden layer delta
        # range hidden layer for tatoal layer -2 to 0 step -1
        for layer_idx in range(len(self.cells) - 2, -1, -1):
            delta_j = []
            for cell_idx, cell_j in enumerate(self.cells[layer_idx]):
                #TODO check delta list
                sigma_dk_wkj = 0
                # print("delta", delta)
                for cell_k, each_delta_k in zip(self.cells[layer_idx + 1], delta[0]):
                    # print(layer_idx, ", ", cell_idx,  ", ", len(self.cells[layer_idx]))
                    # print("idx", cell_idx + 1)
                    # print("weight", len(cell_k.get_weights()))
                    sigma_dk_wkj += each_delta_k * cell_k.get_weights()[cell_idx + 1]
                delta_j.append(cell_j.get_last_y() * (1 - cell_j.get_last_y()) * sigma_dk_wkj)
            # print("delta_j", delta_j)
            #delta use revere method to memerize
            delta.append(delta_j)
            # print("last_delta\t", delta)
        # print("delta\t", delta)

        #adjust weights
        for layer_idx in range(len(self.cells)):
            for cell_idx in range(len(self.cells[layer_idx])):
                self.cells[layer_idx][cell_idx].feedback(delta[len(self.cells) - 1 - layer_idx][cell_idx])

        # for layer_idx in range(len(self.cells)):
        #     for cell_idx in range(len(self.cells[layer_idx])):
        #         print(layer_idx, ", ", cell_idx, "->\t", self.cells[layer_idx][cell_idx].get_weights())

    def get_weights(self):
        return self.cells[-1][0].get_weights()

    def testing(self, testting_set):
        points = testting_set[0]
        ys = testting_set[1]

        suc_point = []
        suc_y = []

        err_point = []
        err_y = []

        min_c = self.class_middle[0]
        for point, y in zip(points, ys):
            out = self.__forward(point)
            # TODO mutilp
            out = out[0]
            # print()

            for c in self.class_middle:
                if abs(out - min_c) > abs(out - c):
                    min_c = c
            # print("ds, ys, jy\t", y, out, min_c)
            if y != min_c:
                err_point.append([self.cells[-2][0].get_last_y(), self.cells[-2][1].get_last_y()])
                err_y.append(y)
            else:
                suc_point.append([self.cells[-2][0].get_last_y(), self.cells[-2][1].get_last_y()])
                suc_y.append(y)

        return (suc_point, suc_y), (err_point, err_y)
    def get_err_rate(self, testing_set):
        s,e = self.testing(testing_set)
        return len(e[0])/(len(e[0]) + len(s[0]))

class cell_sigmoid(cell):
    def __init__(self, dimensions = 2, learning_rate = 0.5, lr_coefficent = 1):
        super().__init__(dimensions = dimensions, learning_rate = learning_rate, lr_coefficent = lr_coefficent)
    def weight_init(self):
        tmp = [-1]
        for i in range(0, self.dimensions):
            tmp.append(random.random() * (1 if i % 2 == 0 else -1))
        self.weights = np.array(tmp)
    def set_weights(self, w):
        self.weights = np.array(w)

    def activation(self):
        stimulates = self.summation()
        #print("stimulation = ", stimulates)
        self.y = 1 / (1 + math.exp(-stimulates))

        # status
        # print("weight ->\t", self.weights)
        # print("input ->\t", self.xs)
        # print("output ->\t", self.y)

        return self.y
    def feedback(self, delta):
        # if self.y < 0:
        #     self.weights = self.weights + self.xs * self.learning_rate
        #         #print("after < ", self.weights , ", ", self.xs)
        # elif self.y >= 0:
        #     self.weights = self.weights - self.xs * self.learning_rate

        delta = np.array(delta)

        # if sum(delta * self.xs) > 0:
        #     if self.dirrection < 0:
        #         self.learning_rate = self.learning_rate * self.lr_coefficent
        #     else:
        #         self.learning_rate = self.learning_rate / self.lr_coefficent
        #     self.dirrection = 1
        # elif sum(delta * self.xs) < 0:
        #     if self.dirrection > 0:
        #         self.learning_rate = self.learning_rate * self.lr_coefficent
        #     else:
        #         self.learning_rate = self.learning_rate / self.lr_coefficent
        #     self.dirrection == -1


        self.weights = self.weights + self.learning_rate * delta * self.xs

        # status
        # print("feedback ->\t", self.weights)
        # print("delta->\t", delta)

class perceptron:
    def __init__(self, dimension = 2, learning_rate = 0.8, training_times = 10, judge = 1, best_w = False):
        self.cells = cell(dimension, learning_rate)
        self.learning_rate = learning_rate
        self.training_times = training_times
        self.dimension = dimension
        self.err_rate = 0
        self.judge = judge
        self.itimes = 0
        self.best_w = best_w
        self.best_d = [0,0]
    def training(self, training_set):
        points = training_set[0]
        ys = training_set[1]
        self.itimes = 0
        for i in range(0, self.training_times):
            x_flag = True
            for point, y in zip(points, ys):
                #print("test point", test)
                x = self.cells.stimulation(point)
                #print("feedback-> ",x, (1 if test[-1] == 2 else -1), test[-1])
                if x != (1 if y == self.judge else -1):
                    self.cells.feedback()
                    x_flag = False
                    if self.best_w:
                        err_rate = self.get_err_rate(training_set)
                        if 1 - err_rate > self.best_d[1]:
                            self.best_d = [i, 1 - err_rate]
                    #print("triger feedback")
                #else:
                    #print(test, ", ",self.cells.y)
                    #print("Success sort")
                #input("pasue")
            self.itimes += 1
            if x_flag:
                break
        print(self.cells.weights)
    def get_weights(self):
        return self.cells.getweights()
    def get_itimes(self):
        return self.itimes
    def get_best_result(self):
        return self.best_d
    def get_err_rate(self, testing_set):
        # print(testing_set)
        count = 0.0
        total = 0.0
        points = testing_set[0]
        ys = testing_set[1]
        # print("test", points, ys)
        for point, y in zip(points, ys):
            #print(input)
            total += 1.0
            #print("test_set, ", point, y)
            if self.cells.stimulation(point) != (1 if y == self.judge else -1):
                count += 1.0
        self.err_rate = count / total
        return self.err_rate

    def test(self, point):
        return self.cells.stimulation(point[0:-1])

def main():
    print("Percetpron Test!!!-------------------------------------------------------------------------!")
    # data = dp("./dataset/2Hcircle1.txt")
    # data = get_data("./dataset/2CloseS3.txt")
    # data = dp("./dataset/C3D.TXT")
    # data = [[0,0,1],[0,1,1],[1,0,-1],[1,1,1]]
    data = [[[1,1],[0,0],[1,0],[0,1]],[0,0,1,1]]
    nn = mlp()
    nn.test_train(data)
    # test = cell_sigmoid(5)
    # tmp = test.stimulation([1,2,-3,-4,-5])
    # print(tmp)


    # data.open_file()
    # data.get_data()
    # print(data.get_data())
    #
    #
    # ne = perceptron(dimension = 2, learning_rate = 0.4, training_times = 30, judge = 1, best_w = False)
    # ne.training(tmp)
    # print("Err Rate: ", ne.get_err_rate(tmp) * 100)


if __name__ == '__main__':
    main()
