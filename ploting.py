#!/usr/bin/env python3
from gi.repository import Gtk
from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as FigureCanvas
from data_proc import data_proc as dp
from gi.repository.GdkPixbuf import Pixbuf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
#draw 2d: points, lines, axises
class paper:
    def __init__(self, unit_xy = (10, 10)):
        self.color = ["r","b","g","c","m","y","k","w"]
        self.data = None
        # self.fig = plt.figure()

        self.fig = Figure(figsize = unit_xy, dpi=100)
        self.fig.patch.set_facecolor('none')

        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.resetpaper()

    def resetpaper(self):
        self.ax.cla()
        self.ax.patch.set_alpha(0)
        # self.ax.set_xlim(0,10)
        # self.ax.set_ylim(0,10)
        self.ax.grid(True)

    def draw(self):
        self.fig.canvas.draw()

        # self.fig.canvas.draw()
        # plt.draw()
        # plt.show()

    def draw_2d_point(self, dataset, class_middle, shape = '.'):
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

        for idx in range(len(class_list)):
            self.ax.plot(tmp_point[idx][0], tmp_point[idx][1], self.color[class_middle.index(class_list[idx])] + shape)
    #draw (point,color, shape)

    #TODO
    def draw_2d_line(self, slope, bias):
        pass
        # if self.weights == 1:
        #     x = np.linspace(plt.xlim()[0], plt.xlim()[1], 2)
        #     y = np.linspace(plt.ylim()[0], plt.ylim()[1], 2)
        #     if self.weights[1] == 0:
        #         y = [self.weights[0]/self.weights[2],self.weights[0]/self.weights[2]]
        #     elif self.weights[2] == 0:
        #         x = [self.weights[0]/self.weights[1],self.weights[0]/self.weights[1]]
        #     else:
        #         #x = [0,self.weights[0]/self.weights[1]]
        #         print("draw")
        #         if -self.weights[1]/self.weights[2] > 1:
        #             print(" > ")
        #             x = (self.weights[0] - self.weights[2] * x) / self.weights[1]
        #         else:
        #             y = (self.weights[0] - self.weights[1] * x) / self.weights[2]
        #     plt.plot(x, y ,"k--")
        # else:
        #     for idx, each_weights in enumerate(self.weights):
        #         print(each_weights)
        #         x = np.linspace(plt.xlim()[0], plt.xlim()[1], 2)
        #         y = np.linspace(plt.ylim()[0], plt.ylim()[1], 2)
        #         if each_weights[1] == 0:
        #             y = [each_weights[0]/each_weights[2],each_weights[0]/each_weights[2]]
        #         elif each_weights[2] == 0:
        #             x = [each_weights[0]/each_weights[1],each_weights[0]/each_weights[1]]
        #         else:
        #             print("normal")
        #             #x = [0,self.weights[0]/self.weights[1]]
        #             if abs(-each_weights[1]/each_weights[2]) > 1:
        #                 x = (each_weights[0] - each_weights[2] * y) / each_weights[1]
        #             else:
        #                 y = (each_weights[0] - each_weights[1] * x) / each_weights[2]
        #         plt.plot(x, y ,color[idx] + "--")

    def __draw_3d(self):
        pass

class MainClass():
    def __init__(self, canvas):
        self.window = Gtk.Window()
        self.window.set_default_size(800, 500)
        self.boxvertical = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.window.add(self.boxvertical)

        # drawing fig
        self.box = Gtk.Box()
        self.boxvertical.pack_start(self.box, True, True, 0)
        self.box.pack_start(canvas, True, True, 0)
        # self.fig.canvas.draw()



def main():
    print("drawing function")
    data = dp('/home/shaowu/code/neural_network/dataset/2Ccircle1.txt')
    data.open_file()
    p = paper()
    p.draw_2d_point(data.get_data())
    p.draw()

    mc = MainClass(p.canvas)

    mc.window.connect("delete-event", Gtk.main_quit)
    mc.window.show_all()
    Gtk.main()


if __name__ == '__main__':
    main()
