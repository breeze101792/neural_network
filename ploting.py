#!/usr/bin/env python3
from data_proc import data_proc as dp
from gi.repository.GdkPixbuf import Pixbuf
import matplotlib.pyplot as plt
import numpy as np
#draw 2d: points, lines, axises
class paper:
    def __init__(self, pix_buf = None):
        self.color = ["r","b","g","c","m","y","k","w"]
        self.data = None
        self.fig = plt.figure()
        self.pix_buf = pix_buf
    def drawing(self):
        fig = plt.figure(figsize=(100,100),dpi=75)
        ax1 = fig.add_axes([0.1, 0.1, 0.4, 0.7])
        ax2 = fig.add_axes([0.55, 0.1, 0.4, 0.7])
        x = np.arange(0.0, 2.0, 0.02)
        y1 = np.sin(2*np.pi*x)
        y2 = np.exp(-x)
        l1, l2 = ax1.plot(x, y1, 'rs-', x, y2, 'go')

        y3 = np.sin(4*np.pi*x)
        y4 = np.exp(-2*x)
        l3, l4 = ax2.plot(x, y3, 'yd-', x, y4, 'k^')

        fig.legend((l1, l2), ('Line 1', 'Line 2'), 'upper left')
        # fig.legend((l3, l4), ('Line 3', 'Line 4'), 'upper right')
        # plt.show()
        print(fig.get_figure())
        self.pix_buf = fig.get_figure()
    # point = [[x1, y1], [x2, y2], ......., [class]]
    def draw_2d_point(self, dataset):
        data_point = dataset[0]
        data_y = dataset[1]
        tmp_point = []
        class_list = []
        for point, y in zip(data_point, data_y):
            if y not in class_list:
                class_list.append(y)
                tmp_point.append([[point[0]],[point[1]]])
            else:
                tmp_point[class_list.index(y)][0].append(point[0])
                tmp_point[class_list.index(y)][1].append(point[1])
        print(tmp_point)

        for idx in range(len(class_list)):
            plt.plot(tmp_point[idx][0], tmp_point[idx][1], self.color[idx] + "o")
        plt.draw()
        plt.show()


    def __draw_2d(self):
        plt.ion()
        points = self.training_set[0]
        ys = self.training_set[1]
        for point, y in zip(points, ys):
            if y == 0:
                plt.plot(point[0], point[1] ,color[0] + "o")
            elif y == 1:
                plt.plot(point[0], point[1] ,color[1] + "o")
            elif y == 2:
                plt.plot(point[0], point[1] ,color[2] + "o")
            elif point[-1] == 3:
                plt.plot(point[0], point[1] ,color[3] + "o")
            elif point[-1] == 4:
                plt.plot(point[0], point[1] ,color[4] + "o")
            elif point[-1] == 5:
                plt.plot(point[0], point[1] ,color[5] + "o")
            elif point[-1] == 6:
                plt.plot(point[0], point[1] ,color[6] + "o")

        #plt.plot([2,0,-2,0], [0,2,0,-2] ,"wo")
        print(self.weights)
        if self.weights == 1:
            x = np.linspace(plt.xlim()[0], plt.xlim()[1], 2)
            y = np.linspace(plt.ylim()[0], plt.ylim()[1], 2)
            if self.weights[1] == 0:
                y = [self.weights[0]/self.weights[2],self.weights[0]/self.weights[2]]
            elif self.weights[2] == 0:
                x = [self.weights[0]/self.weights[1],self.weights[0]/self.weights[1]]
            else:
                #x = [0,self.weights[0]/self.weights[1]]
                print("draw")
                if -self.weights[1]/self.weights[2] > 1:
                    print(" > ")
                    x = (self.weights[0] - self.weights[2] * x) / self.weights[1]
                else:
                    y = (self.weights[0] - self.weights[1] * x) / self.weights[2]
            plt.plot(x, y ,"k--")
        else:
            for idx, each_weights in enumerate(self.weights):
                print(each_weights)
                x = np.linspace(plt.xlim()[0], plt.xlim()[1], 2)
                y = np.linspace(plt.ylim()[0], plt.ylim()[1], 2)
                if each_weights[1] == 0:
                    y = [each_weights[0]/each_weights[2],each_weights[0]/each_weights[2]]
                elif each_weights[2] == 0:
                    x = [each_weights[0]/each_weights[1],each_weights[0]/each_weights[1]]
                else:
                    print("normal")
                    #x = [0,self.weights[0]/self.weights[1]]
                    if abs(-each_weights[1]/each_weights[2]) > 1:
                        x = (each_weights[0] - each_weights[2] * y) / each_weights[1]
                    else:
                        y = (each_weights[0] - each_weights[1] * x) / each_weights[2]
                plt.plot(x, y ,color[idx] + "--")
        #print(x, ", ",y)



        plt.draw()
        plt.show()
    def __draw_3d(self):
        pass






def main():
    print("drawing function")
    data = dp('/home/shaowu/code/neural_network/dataset/2Ccircle1.txt')
    data.open_file()
    buf = None
    p = paper(buf)
    print(data.get_data())
    p.draw_2d_point(data.get_data())

if __name__ == '__main__':
    main()
