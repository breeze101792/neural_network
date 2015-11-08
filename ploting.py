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
    def __init__(self, pix_buf = None):
        self.color = ["r","b","g","c","m","y","k","w"]
        self.data = None
        # self.fig = plt.figure()
        self.pix_buf = pix_buf

        self.fig = Figure(figsize=(10,10), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

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

        # fig.legend((l1), ("test"), 'upper right')
        # fig.legend((l3, l4), ('Line 3', 'Line 4'), 'upper right')
        plt.show()
        print(fig.get_figure())
    def resetpaper(self):
        self.ax.cla()
        # self.ax.set_xlim(0,10)
        # self.ax.set_ylim(0,10)
        self.ax.grid(True)

    def draw(self):
        pass
        # self.fig.canvas.draw()
        # plt.draw()
        # plt.show()

    def draw_2d_point(self, dataset):
        self.resetpaper()
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
            self.ax.plot(tmp_point[idx][0], tmp_point[idx][1], self.color[idx] + "o")
        self.fig.canvas.draw()

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
