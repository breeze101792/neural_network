#!/usr/bin/env python3
import random
import numpy as np
class myDict(dict):

    def __init__(self):
        self = dict()

    def add(self, key, value):
        self[key] = value

class data_proc:
    def __init__(self, file_name = '/home/shaowu/code/neural_network/dataset/2Ccircle1.txt'):
        self.file_name = file_name
        self.data = []
        self.ys = []
        self.nom_ys = []
        self.class_table = []
    def set_file_name(self, file_name):
        self.file_name = file_name
    def open_file(self):
        self.data = []
        self.ys = []
        self.nom_ys = []
        self.class_table = []

        tmp_table = myDict()
        class_num = 0
        with open(self.file_name) as f:
            for line in f:
                # print([float(x) for x in line.split()])
                self.data.append([float(x) for x in line.split()[0:-1]])
                self.ys.append(float(line.split()[-1]))
        # print(self.data)
    @staticmethod
    def to_ndata(points):
        tmp = []
        for p in points:
            tmp.append(np.array(p))
            print(p)
        return tmp

    def get_data(self, rate_of_data = 1, is_random = True, approach = "class"):
        self.__data_normalize(approach)
        tmp_data = self.data.copy()
        tmp_nom_ys = self.nom_ys.copy()
        sub_dataset = []
        sub_ys = []
        tmp = 0
        # print(self.data, self.nom_ys)
        #print(rate_of_data * len(self.data))
        if is_random:
            for _ in range(int(rate_of_data * len(self.data))):
                tmp = random.randint(0, len(tmp_data))
                sub_dataset.append(tmp_data.pop(tmp - 1))
                sub_ys.append(tmp_nom_ys.pop(tmp - 1))
                #print(len(self.data))
            #print(self.data)
            return sub_dataset, sub_ys
        elif not is_random:
            return self.data, self.nom_ys
    def get_another_data(self, part_of_data):
        sub_dataset = []
        sub_ys = []
        for point, y in zip(self.data, self.ys):
            if point not in part_of_data:
                sub_dataset.append(point)
                sub_ys.append(y)
        return sub_dataset, sub_ys
    #approach = class/function
    def __data_normalize(self, approach = "class"):
        if approach == "class":
            print("class!")
            self.class_table = myDict()
            self.nom_ys = []
            class_num = 0
            for each_y in self.ys:
                if each_y not in self.class_table.values():
                    self.class_table.add(class_num, each_y)
                    self.nom_ys.append(class_num)
                    class_num += 1
                    # print("add ", self.class_table)
                else:
                    self.nom_ys.append([k for k, v in self.class_table.items() if v == each_y][0])
                    #print([k for k, v in self.class_table.items() if v == each_y])
                    #print("no add ", self.class_table)
        elif approach == "function":
            print("function")

def main():
    print("dataset class test")
    data = data_proc()
    data.open_file()
    data.get_data()#is_random = True, rate_of_data = 0.5)
    # print(data.get_data())
    print(data_proc.to_ndata(data.get_data()[0]))
    #print(data.nom_ys)
    #print(data.data, "\n", data.class_table, "\n", data.ys, "\n", data.get_data(is_random = True))
    #print(len(data.ys), ", ", len(data.data))

if __name__ == '__main__':
    main()
