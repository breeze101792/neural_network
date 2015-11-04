#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from data_proc import data_proc as dp

class cell:
    def __init__(self, dimensions = 2, learning_rate = 0.4):
        self.dimensions = dimensions
        self.learning_rate = learning_rate
        self.xs = 0
        self.y = 0
        self.weights = None
        self.in_cells = []
        self.out_cells = []
        self.weight_init()
        print("dim -> ", dimensions)
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
    def get_last_y(self):
        return self.y
    def __str__(self):
        print("test")

class mlp:
    def __init__(self, structure = [2], dimension = 4, learning_rate = 0.4, numofclass = 2, training_times = 10, judge = 1, best_w = False):
        self.cells = []
        self.learning_rate = learning_rate
        self.training_times = training_times
        self.dimension = dimension
        self.err_rate = 0
        self.judge = judge
        self.itimes = 0
        self.best_w = best_w
        self.best_d = [0,0]

        # structure is [num of lay1, num of lay2, ...., num of layp]
        self.structure = structure
        self.numofhiddenlayer = len(structure)
        self.numofoutput = 1
        self.numofclass = numofclass
        self.struct_init()
    def struct_init(self):
        self.cells = []
        tmp_cell = []
        num_of_inputs = self.dimension

        for lay in range(self.numofhiddenlayer):
            tmp_cell = []
            for cell in range(self.structure[lay]):
                tmp_cell.append(cell_sigmoid(num_of_inputs, self.learning_rate))
            num_of_inputs = self.structure[lay]
            self.cells.append(tmp_cell)
        tmp_cell = []
        for cell in range(self.numofoutput):
            tmp_cell.append(cell_sigmoid(num_of_inputs, self.learning_rate))
        self.cells.append(tmp_cell)
        # print(self.cells)
    def training(self, training_set):
        self.cells[0][0].set_weights([-1.2,1,1])
        self.cells[0][1].set_weights([0.3,1,1])
        self.cells[1][0].set_weights([0.5,0.4,0.8])
        print("forward ->\t", self.__forward([1,1]))
        self.__backward([0])

        pass

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
    def __backward(self, ys):
        delta_j = []
        delta_k = []
        cell_y = 0

        # Adjust weight for output layer
        for idx, cell_j in enumerate(self.cells[-1]):
            # calc delta value
            cell_y = cell_j.get_last_y()
            delta_k.append((ys[idx] - cell_y) * cell_y * (1 - cell_y))
            print(idx, ", ", delta_k)
            # Adjust weight
            cell_j.feedback(delta_k[-1])

        # Adjust weight for hidden layers
        # for each layer
        for layer in range(self.numofhiddenlayer - 1, -1,-1):
            # for each cell in the layer
            # k mean previous layer
            for cell_j in self.cells[layer]:
                weight_kj = []
                sum_wd = 0
                # calc sum of delta_j * weight_kj
                for idx, cell_k in enumerate(self.cells[layer + 1]):
                    # idx + 1, 1 for bias X0
                    weight_kj.append(cell_k.getweights()[idx + 1])
                    print(delta_k, ", ", idx)
                    sum_wd += weight_kj[-1] * delta_k[idx]
                # calc delta_j
                delta_j.append(cell_j.get_last_y() * (1 - cell_j.get_last_y()) * sum_wd)
                cell_j.feedback(delta_j[-1])
                # for next loopt as it's next layer
                delta_k = delta_j

class cell_sigmoid(cell):
    def __init__(self, dimensions = 2, learning_rate = 0.5):
        super().__init__(dimensions = dimensions, learning_rate = learning_rate)
    def weight_init(self):
        tmp = [-1]
        for i in range(0, self.dimensions):
            tmp.append(random.random())
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
        self.weights = self.weights + self.learning_rate * delta * self.xs

        # status
        print("feedback ->\t", self.weights)

class perceptron:
    def __init__(self, dimension = 2, learning_rate = 0.4, training_times = 10, judge = 1, best_w = False):
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
    print("Percetpron Test!!!")
    data = dp("./dataset/2Hcircle1.txt")
    #data = get_data("./dataset/2CloseS3.txt")
    # data = dp("./dataset/C3D.TXT")
    #data = [[0,0,1],[0,1,1],[1,0,-1],[1,1,1]]

    nn = mlp()
    nn.training([1])
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
