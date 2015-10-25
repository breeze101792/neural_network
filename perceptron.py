#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import math
'''
def extended(ax, x, y, **args):

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x_ext = np.linspace(xlim[0], xlim[1], 100)
    p = np.polyfit(x, y , deg=1)
    y_ext = np.poly1d(p)(x_ext)
    ax.plot(x_ext, y_ext, **args)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax
'''
class cell:
    def __init__(self, num_in = 2, learning_rate = 0.4):
        self.num_in = num_in
        self.learning_rate = learning_rate
        self.xs = 0
        self.y = 0
        self.weights = [-1]
        self.in_cells = []
        self.out_cells = []
        for i in range(0, self.num_in):
            self.weights.append(0 if i % 2 == 0 else 1)
        #print("hello")
    def summation(self):
        x = 0
        for i in range(0, self.num_in + 1):
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
                    err_rate = self.get_err_rate(training_set)
                    if self.best_w:
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
        count = 0.0
        total = 0.0
        points = testing_set[0]
        ys = testing_set[1]
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
'''
class perceptron_single:
    def __init__(self, data, class_num, learning_rate = 0.4, training_times = 10, best_w = False):
        self.class_num = class_num
        self.data = data
        self.perceptrons = []
        self.weights = []
        self.learning_rate = learning_rate
        self.training_times = training_times
        self.best_w = best_w
    def training(self):
        for i in range(self.class_num):
            self.perceptrons.append(perceptron(self.data, dimension = 2, learning_rate = self.learning_rate, training_times = self.training_times, judge = i, best_w = self.best_w))
            self.perceptrons[i].training()
            self.weights.append(self.perceptrons[i].get_weights())
    def get_weights(self):
        return self.weights
    def get_itimes(self):
        x = 0
        for idx, i in enumerate(self.perceptrons):
            if x < i.get_itimes():
                 x = idx
        return self.perceptrons[idx].get_itimes()
    def get_best_result(self):
        x = 0
        for idx,i in enumerate(self.perceptrons):
            if x < i.best_d[1]:
                 x = idx
        return self.perceptrons[idx].best_d
    def get_err_rate(self):
        count = 0
        number = 0
        for point in self.data:
            ans = True
            print("new point", point)
            for idx, perc in enumerate(self.perceptrons):
                x = perc.test(point)
                print(ans, ", ", x, ", ", idx)
                if ans == True and x == 1 and idx != point[-1]:
                    ans = False
                elif ans == True and x == -1 and idx == point[-1]:
                    ans == False
            number += 1
            if ans == True:
                count += 1
        return 1 - count / number

class cell_sigmoid(cell):
    def __init__(self, num_in = 2, learning_rate = 0.4, v_para = 1):
        super().__init__(num_in = 2, learning_rate = 0.4, v_para = 1)
        self.v_para = v_para
    def activation(self):
        stimulates = self.summation()
        #print("stimulation = ", stimulates)
        self.y = (1 - math.e ** (-2 * self.v_para *  stimulates))/(1 + math.e ** (-2 * self.v_para * stimulates))
        return self.y

class perceptron_m:
    def __init__(self, data, num_classes = 1, dimension = 2, learning_rate = 0.4, training_times = 10):
        self.cells = []
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.training_times = training_times
        self.data = data
        self.dimension = dimension
        self.err_rate = 0
        for _ in range(0, num_classes):
            self.cells.append(cell_sigmoid(dimension, learning_rate))

    def training(self):
        for _ in range(0, self.training_times):
            for test in self.data:
                for idx, each_cell in enumerate(self.cells, start=0):

                    print(idx, ":test point", test)
                    x = each_cell.stimulation(test[0:-1])
                    print("feedback-> ",x, test[-1])
                    if idx == test[-1] and x > 0:
                        each_cell.feedback()
                    elif idx != test[-1] and x <= 0:
                        each_cell.feedback()
                    else:
                        #print(test, ", ",each_cell.y)
                        print("Success sort")
    def get_weights(self):
        pass
        return self.cells.getweights()
    def get_err_rate(self):
        pass
        count = 0.0
        total = 0.0

        for test in self.data:
            total += 1
            prob_c = -1
            prob_max = -1
            for idx, each_cell in enumerate(self.cells, start=0):
                x = each_cell.stimulation(test[0:-1])
                if idx == 0 or prob_max < x:
                    prob_c = idx
                    prob_max = x
            if test[-1] == prob_c:
                count += 1
        self.err_rate = count / total
        return self.err_rate
'''

def get_data(fn):
    data = []
    with open(fn) as f:
        for line in f:
            data.append([float(num) for num in line.split()])
    return data

def main():
    print("Percetpron Test!!!")
    #data = get_data("./dataset/2Hcircle1.txt")
    #data = get_data("./dataset/2CloseS3.txt")
    data = get_data("./dataset/C3D.TXT")
    #data = [[0,0,1],[0,1,1],[1,0,-1],[1,1,1]]
    '''
    ne = perceptron(data,3, 0.2, 100)
    ne.training()
    print("Err Rate: ", ne.get_err_rate() * 100)
    '''
    mnn = perceptron_m(data, 4, 3, training_times = 1)
    mnn.training()
    print("Err Rate: ", mnn.get_err_rate() * 100)


if __name__ == '__main__':
    main()
