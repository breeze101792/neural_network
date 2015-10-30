#!/usr/bin/env python3
from data_proc import data_proc as dp
#draw 2d: points, lines, axises
class paper:
    def __init__(self, type = 2):
        self.color = ["r","b","g","c","m","y","k","w"]
        self.data = None
        self.type = type
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

if __name__ == '__main__':
    main()
