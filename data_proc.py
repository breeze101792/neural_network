#!/usr/bin/env python3
import random
import numpy as np
class myDict(dict):

    def __init__(self):
        self = dict()

    def add(self, key, value):
        self[key] = value

class data_proc:
    def __init__(self, file_name = None, raw_output = 0):
        self.file_name = file_name
        self.data = []
        self.ys = []
        self.nom_ys = []
        self.class_table = []
        self.class_num = 0
        self.max_min = None
        self.raw_output = raw_output
    def set_file_name(self, file_name):
        self.file_name = file_name
    def open_file(self):
        self.data = []
        self.ys = []
        self.nom_data = []
        self.nom_ys = []
        self.class_table = []
        self.max_min = []

        tmp_table = myDict()
        self.class_num = 0
        tmp_list = None
        with open(self.file_name) as f:
            for line in f:
                tmp_list = line.split()
                #init max_min for data normalize
                if self.max_min == []:
                    # ([min],[max])
                    for each_d in range(len(tmp_list)):
                        self.max_min.append([float(tmp_list[each_d]), float(tmp_list[each_d])])

                if float(tmp_list[-1]) not in self.ys:
                    self.class_num += 1

                # print([float(x) for x in line.split()])
                for idx_d, each_x in enumerate(tmp_list[0:len(tmp_list)]):
                    each_x = float(each_x)
                    # print("mm a\t", self.max_min[idx_d])
                    # print("each_x\t", each_x)
                    if self.max_min[idx_d][0] > each_x:
                        self.max_min[idx_d][0] = each_x
                    if self.max_min[idx_d][1] < each_x:
                        self.max_min[idx_d][1] = each_x
                    # print("mm b\t", self.max_min[idx_d])

                self.data.append([float(x) for x in tmp_list[0:-1]])
                self.ys.append(float(tmp_list[-1]))
        # print(self.data)
    @staticmethod
    def to_ndata(points):
        tmp = []
        for p in points:
            tmp.append(np.array(p))
            # print(p)
        return tmp
    def get_class_middle(self):
        return list(self.class_table.keys())
    def get_data(self, rate_of_data = 1, is_random = True, approach = "class"):
        self.__data_normalize(approach)
        tmp_data = self.nom_data.copy()
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
            return self.nom_data, self.nom_ys
    def get_another_ndata(self, part_of_data):
        sub_dataset = []
        sub_ys = []
        for point, y in zip(self.nom_data, self.nom_ys):
            if point not in part_of_data:
                sub_dataset.append(point)
                sub_ys.append(y)
        return sub_dataset, sub_ys
    #approach = class/function
    def __data_normalize(self, approach = "class"):
        if approach == "class":
            print("class method")
            self.class_table = myDict()
            self.nom_data = []
            self.nom_ys = []
            distance = 1 / (self.class_num * 2)
            class_middle = distance
            # print("max_min\t", self.max_min)
            for idx_data in range(len(self.data)):
                # print("data\t", self.data[idx_data])
                tmp_data = []
                for idx_dim in range(len(self.data[idx_data])):
                    tmp_data.append((self.data[idx_data][idx_dim] - self.max_min[idx_dim][0]) / (self.max_min[idx_dim][1] - self.max_min[idx_dim][0]))
                    # print("tmpe_data\t", (self.data[idx_data][idx_dim] - self.max_min[idx_dim][0]) / self.max_min[idx_dim][1])
                self.nom_data.append(tmp_data)
            # print(self.nom_data)

            for each_y in self.ys:
                if each_y not in self.class_table.values():
                    self.class_table.add(class_middle, each_y)
                    self.nom_ys.append(class_middle)
                    class_middle += distance * 2
                    # print("add ", self.class_table)
                else:
                    self.nom_ys.append([k for k, v in self.class_table.items() if v == each_y][0])
                    #print([k for k, v in self.class_table.items() if v == each_y])
                    #print("no add ", self.class_table)
            # print("class table\t", self.class_table)
        elif approach == "func":
            # TODO fix range
            print("func")
            self.class_table = myDict()
            self.nom_data = []
            self.nom_ys = []
            # print("max_min\t", self.max_min)
            for idx_data in range(len(self.data)):
                # print("data\t", self.data[idx_data])
                tmp_data = []
                for idx_dim in range(len(self.data[idx_data])):
                    tmp_data.append((self.data[idx_data][idx_dim] - 0) / (30 - 0))
                    # print("tmpe_data\t", (self.data[idx_data][idx_dim] - self.max_min[idx_dim][0]) / self.max_min[idx_dim][1])
                self.nom_data.append(tmp_data)
            # print(self.nom_data)

            for each_y in self.ys:
                self.nom_ys.append(((each_y + 40)/ 80))

                # if each_y not in self.class_table.values():
                #     # self.nom_ys.append((each_y - self.max_min[-1][0]) / (self.max_min[-1][1] - self.max_min[-1][0]))
                #     self.nom_ys.append((each_y - self.max_min[-1][0]) / (self.max_min[-1][1] - self.max_min[-1][0]))
                #     # print("add ", self.class_table)
                # else:
                #     self.nom_ys.append([k for k, v in self.class_table.items() if v == each_y][0])
                    #print([k for k, v in self.class_table.items() if v == each_y])
                    #print("no add ", self.class_table)
            # print("class table\t", self.class_table)
        else:
            print("old")
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
    def get_data_size(self):
        return len(self.data)
    def get_data_classification_num(self):
        return len(self.class_table)
    def get_data_dimension(self):
        return len(self.data[0])
def main():
    print("dataset class test")
    data = data_proc('/home/shaowu/code/neural_network/dataset/2Ccircle1.txt')
    data.open_file()
    print(data.class_num)
    print(data.get_data_size())
    # print(data_proc.to_ndata(data.get_data()[0]))
    #print(data.nom_ys)
    #print(data.data, "\n", data.class_table, "\n", data.ys, "\n", data.get_data(is_random = True))
    #print(len(data.ys), ", ", len(data.data))

if __name__ == '__main__':
    main()
