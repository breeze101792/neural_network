#!/usr/bin/env python3
from gi.repository import Gtk
from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as FigureCanvas
from data_proc import data_proc as dp
from gi.repository.GdkPixbuf import Pixbuf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from math import *
#draw 2d: points, lines, axises
class Car:
    def __init__(self):
        self.sensor_left = 0
        self.sensor_middle = 0
        self.sensor_right = 0
        self.pos = [0,0]
        self.dir = 1.5700
        self.radius = 3
        self.track = []
    def step_forward(self, theta):
        # self.pos = [self.pos[0] - cos(self.dir), self.pos[1] + sin(self.dir)]
        self.pos = [self.pos[0] + cos(self.dir + theta) + sin(theta)*sin(self.dir), self.pos[1] + sin(self.dir + theta) + sin(theta)*sin(self.dir)]
        self.track.append(self.pos)
        self.dir = self.dir - np.arcsin(2 * sin(theta) / 3)
        if self.dir < 0:
            self.dir = self.dir + 6.2831853072
        elif self.dir > 6.2831853072:
            self.dir = self.dir - 6.2831853072

    def set_dir(self, dir):
        self.dir = dir

class racing_map:
    def __init__(self, unit_xy = (10, 10), size = ((-8, 32), (-2, 40)), title = ""):
        self.color = ["r","b","g","c","m","y","k","w"]
        self.map_data = (((-6,0), (-6,22)), ((-6,22), (18,22)), ((18,22), (18,37)), ((6,0), (6,10)), ((6,10), (30,10)), ((30,10), (30,37)))
        self.data = None
        self.map_size = size
        # self.fig = plt.figure()

        self.fig = Figure(figsize = unit_xy, dpi=80)
        self.fig.patch.set_facecolor('none')

        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.title = title
        self.resetpaper()

    def resetpaper(self):
        self.ax.cla()
        self.ax.patch.set_alpha(0)
        self.ax.set_title(self.title)
        self.expend_lim(self.map_size)
        self.draw_map()
        # self.ax.set_xlim(-0.02,1.02)
        # self.ax.set_ylim(-0.02,1.02)
        # self.ax.grid(True)
    def expend_lim(self, size):
        self.ax.set_xlim(size[0][0], size[0][1])
        self.ax.set_ylim(size[1][0], size[1][1])
    def draw_map(self):
        # map_data = (((-6,0), (-6,22)), ((-6,22), (18,22)), ((18,22), (18,37)), ((6,0), (6,10)), ((6,10), (30,10)), ((30,10), (30,37)))
        for each_line in self.map_data:
            self.ax.plot([each_line[0][0], each_line[1][0]], [each_line[0][1], each_line[1][1]], 'b-')
    def draw_car(self, car):
        circ=plt.Circle(car.pos, radius=car.radius, color='g', fill=False)

        # circ=plt.Circle([0,0], radius=5, color='g', fill=False)
        self.ax.add_patch(circ)
        self.ax.plot([car.pos[0], car.pos[0] + 5 * cos(car.dir)], [car.pos[1], car.pos[1] + 5 * sin(car.dir)], 'g-');
        for point in car.track:
            self.ax.plot(point[0], point[1], 'bo');
    def draw(self, car):
        self.resetpaper()
        self.draw_car(car)
        self.fig.canvas.draw()
    def update_sensor(self, car):
        # print("car -info", car.pos, car.dir)
        car.sensor_middle = self.get_distance(car.pos, car.dir)
        car.sensor_right = self.get_distance(car.pos, car.dir - 0.785398  if car.dir - 0.785398 > 0 else car.dir - 0.785398 + 6.2831853072)
        car.sensor_left = self.get_distance(car.pos, car.dir + 0.785398  if car.dir + 0.785398 < 6.2831853072 else car.dir + 0.785398 - 6.2831853072)
        # self.fig.canvas.draw()
    def test_sensor(self):
        print("result", self.get_distance([0.03341244059388227, 5.057892606615268] ,  0.745918017827))
    def get_distance(self, pos, dir):
        bound = 30
        [min, xd, yd, tmp] = [1600,0,0,0,]
        [tmp_x, tmp_y] = [0,0]
        [x, y] = pos
        # print("pos, dir", pos, ", ", dir)

        if dir <= 1.5707963268:
            # print("if 1")
            xd = 6 - x
            yd = xd * tan(dir)

            # TODO
            tmp = np.power(xd if xd < bound else bound, 2) + np.power(yd if yd < bound else bound, 2)
            if (min > tmp and xd >= 0 and yd >= 0 and y + yd < 10):
                min = tmp
                [tmp_x, tmp_y] = [xd, yd]
                # print('1')

            yd = 22 - y
            xd = yd / tan(dir)


            tmp = np.power(xd if xd < bound else bound, 2) + np.power(yd if yd < bound else bound, 2)
            if (min > tmp and xd >= 0 and yd >= 0 and x + xd <= 18):
                min = tmp
                [tmp_x, tmp_y] = [xd, yd]
                # print('2')

            xd = 30 - x
            yd = xd * tan(dir)


            tmp = np.power(xd if xd < bound else bound, 2) + np.power(yd if yd < bound else bound, 2)
            if (min > tmp and xd >= 0 and yd >= 0):
                min = tmp
                [tmp_x, tmp_y] = [xd, yd]
                # print('3')

            # self.ax.plot([pos[0], pos[0] + tmp_x], [pos[1], pos[1] + tmp_y], 'r-')
        elif dir <= 3.1415926536:
            xd = x - -6
            yd = xd * tan(3.1415926536 - dir)


            tmp = np.power(xd if xd < bound else bound, 2) + np.power(yd if yd < bound else bound, 2)
            if (min > tmp and xd >= 0 and yd >= 0):
                min = tmp
                [tmp_x, tmp_y] = [xd, yd]
                # print('1')

            yd = 22 - y
            xd = yd / tan(3.1415926536 - dir)


            tmp = np.power(xd if xd < bound else bound, 2) + np.power(yd if yd < bound else bound, 2)
            if (min > tmp and xd >= 0 and yd >= 0 and x + xd <= 18):
                min = tmp
                [tmp_x, tmp_y] = [xd, yd]
                # print('2')


            xd = x - 18
            yd = xd * tan(3.1415926536 - dir)


            tmp = np.power(xd if xd < bound else bound, 2) + np.power(yd if yd < bound else bound, 2)
            if (min > tmp and xd >= 0 and yd >= 0 and y + yd >= 22):
                min = tmp
                [tmp_x, tmp_y] = [xd, yd]
                # print('3')

            # self.ax.plot([pos[0], pos[0] - tmp_x], [pos[1], pos[1] + tmp_y], 'g-')
        elif dir <= 4.7123889804:
            xd = x - -6
            yd = xd * tan(dir - 3.1415926536)


            tmp = np.power(xd if xd < bound else bound, 2) + np.power(yd if yd < bound else bound, 2)
            # print(xd, ", ", yd, ", ", tmp)
            if (min > tmp and xd >= 0 and yd >= 0):
                min = tmp
                [tmp_x, tmp_y] = [xd, yd]
                # print('1')

            yd = y - 10
            xd = yd / tan(dir - 3.1415926536)


            tmp = np.power(xd if xd < bound else bound, 2) + np.power(yd if yd < bound else bound, 2)
            if (min > tmp and xd >= 0 and yd >= 0 and x + xd >= 6):
                min = tmp
                [tmp_x, tmp_y] = [xd, yd]
                # print('2')

            xd = x - 18
            yd = xd * tan(dir - 3.1415926536)


            tmp = np.power(xd if xd < bound else bound, 2) + np.power(yd if yd < bound else bound, 2)
            if (min > tmp and xd >= 0 and yd >= 0 and y + yd >= 22):
                min = tmp
                [tmp_x, tmp_y] = [xd, yd]
                # print('3')
            # self.ax.plot([pos[0], pos[0] - tmp_x], [pos[1], pos[1] - tmp_y], 'b-')
        else:
            xd = 6 - x
            yd = xd * tan(6.2831853072 - dir)


            tmp = np.power(xd if xd < bound else bound, 2) + np.power(yd if yd < bound else bound, 2)
            if (min > tmp and xd >= 0 and yd >= 0 and y - yd <= 10):
                min = tmp
                [tmp_x, tmp_y] = [xd, yd]

            yd = y - 10
            xd = yd / tan(6.2831853072 - dir)


            tmp = np.power(xd if xd < bound else bound, 2) + np.power(yd if yd < bound else bound, 2)
            if (min > tmp and xd >= 0 and yd >= 0 and x + xd >= 6):
                min = tmp
                [tmp_x, tmp_y] = [xd, yd]

            xd = 30 - x
            yd = xd * tan(6.2831853072 - dir)


            tmp = np.power(xd if xd < bound else bound, 2) + np.power(yd if yd < bound else bound, 2)
            if (min > tmp and xd >= 0 and yd >= 0):
                min = tmp
                [tmp_x, tmp_y] = [xd, yd]
            # self.ax.plot([pos[0], pos[0] + tmp_x], [pos[1], pos[1] - tmp_y], 'y-')
        # return sqrt(tmp_x * tmp_x + tmp_y * tmp_y)
        tmp = sqrt(tmp_x * tmp_x + tmp_y * tmp_y)


        return (tmp if tmp <= 30 else 30) / 30

















    # TODO i am a line
