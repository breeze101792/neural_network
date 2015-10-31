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
        self.weights = [-1]
        self.in_cells = []
        self.out_cells = []
        self.weight_init()
    def weight_init(self):
        for i in range(0, self.dimensions):
            self.weights.append(0 if i % 2 == 0 else 1)
    def summation(self):
        x = 0
        for i in range(0, self.dimensions + 1):
            x += self.weights[i] * self.xs[i]
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
        #print("before ", self.weights , ", ", self.xs)
        for i in range(0, len(self.weights)):
            if self.y < 0:
                self.weights[i] += self.xs[i] * self.learning_rate
                #print("after < ", self.weights , ", ", self.xs)
            elif self.y >= 0:
                self.weights[i] -= self.xs[i] * self.learning_rate
                #print("after >= ", self.weights , ", ", self.xs)
    def stimulation(self, stimulates):
        self.xs = [-1]
        self.xs.extend(stimulates)
        #print("start", self.weights, self.xs)
        return self.activation()

    def getweights(self):
        return self.weights

class cell_sigmoid(cell):
    def __init__(self, dimensions = 2, learning_rate = 0.5):
        super().__init__(dimensions = dimensions, learning_rate = learning_rate)
    def weight_init(self):
        for i in range(0, self.dimensions):
            self.weights.append(random.random())
    def activation(self):
        stimulates = self.summation()
        #print("stimulation = ", stimulates)
        self.y = (1 - math.e ** (-2 * self.v_para *  stimulates))/(1 + math.e ** (-2 * self.v_para * stimulates))
        return self.y

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

    test = cell_sigmoid()
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
