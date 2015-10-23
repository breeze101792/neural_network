#!/usr/bin/env python3
import random
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


    def open_file(self):
        self.data = []

        tmp_table = myDict()
        class_num = 0
        with open(self.file_name) as f:
            for line in f:
                #print([float(x) for x in line.split()])
                self.data.append([float(x) for x in line.split()[0:-1]])
                self.ys.append(float(line.split()[-1]))
                '''
                tmp_list = []
                for idx, num in enumerate(line.split(), start = 0):
                    if idx == len(line.split()) - 1:
                        if num not in self.class_table.values():
                            self.class_table.add(class_num, num)
                            tmp_table.add(num, class_num)
                            tmp_list.append(int(class_num))
                            class_num += 1
                        else:
                            tmp_list.append(int(tmp_table[num]))
                    else:
                        tmp_list.append(float(num))
                self.data.append(tmp_list)
                '''
        #print(self.data)
    def get_data(self, rate_of_data = 1, is_random = False, approach = "class"):
        self.__data_normalize(approach)
        tmp_data = self.data
        tmp_nom_ys = self.nom_ys
        sub_dataset = []
        sub_ys = []
        if is_random:
            for _ in range(int(rate_of_data * len(self.data))):
                sub_dataset.append(tmp_data.pop(random.randint(0, len(tmp_data) - 1)))
                sub_ys.append(tmp_nom_ys.pop(random.randint(0, len(tmp_nom_ys) - 1)))
            return sub_dataset, sub_ys
        elif not is_random:
            return self.data, self.nom_ys

    #approach = class/function
    def __data_normalize(self, approach = "class"):
        if approach == "class":
            print("class!")
            self.class_table = myDict()
            class_num = 0
            for each_y in self.ys:
                if each_y not in self.class_table.values():
                    self.class_table.add(class_num, each_y)
                    self.nom_ys.append(class_num)
                    class_num += 1
                    #print("add ", self.class_table)
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
    data.get_data(is_random = True, rate_of_data = 0.5)
    #print(data.nom_ys)
    #print(data.data, "\n", data.class_table, "\n", data.ys, "\n", data.get_data(is_random = True))
    #print(len(data.ys), ", ", len(data.data))

if __name__ == '__main__':
    main()
