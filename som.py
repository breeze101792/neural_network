import numpy as np
from ploting import paper
import threading
# for 2 dimension
class SOM:
    def __init__(self, i_len = 100, j_len = 100, sd = 1, learning_rate = 0.4, training_times = 5):
        self.net = None
        self.dimension = 2
        self.i_len = i_len
        self.j_len = j_len
        self.learning_rate = learning_rate
        self.sdp = (sd ** 2) * 2
        self.training_times = training_times
        self.threshold = 0.001
        print(i_len, j_len, sd, learning_rate, training_times)
    def net_init(self):
        self.net = [[] for _ in range(self.i_len)]
        for i in range(self.i_len):
            self.net[i] = [np.array((i / (self.i_len - 1), j / (self.j_len - 1)))for j in range(self.j_len)]
    def whowin(self, point):
        winner = np.array([0,0])
        winner_ip = 100
        cur_ip = 0
        for i in range(self.i_len):
            for j in range(self.j_len):
                cur_ip = np.linalg.norm((point - (self.net[i][j])) if np.sum(self.net[i][j]) != 0 else point)
                if winner_ip > cur_ip:
                    winner_ip = cur_ip
                    winner[0] = i
                    winner[1] = j
        # print("winner", winner[0], ", ",winner[1])
        return winner
        # return np.array([0 ,0])
    def net_update(self, point, winner):
        i_start = j_start = 0
        i_end = self.i_len - 1
        j_end = self.j_len - 1
        tmp = 0
        # print(winner, point)
        for idx in range(winner[0], -1, -1):
            distance = np.linalg.norm(winner - np.array([idx, winner[1]]))
            tmp = np.exp(-(distance ** 2)/self.sdp ) * self.learning_rate
            # print(idx, tmp)
            if tmp < self.threshold:
                # input("press enter")
                i_start= idx
                break

        for idx in range(winner[0], self.i_len):
            distance = np.linalg.norm(winner - np.array([idx, winner[1]]))
            tmp = np.exp(-(distance ** 2)/self.sdp ) * self.learning_rate
            # print(idx, tmp)
            if tmp < self.threshold:
                # input("press enter")
                i_end = idx
                break
        for idx in range(winner[1], -1, -1):
            distance = np.linalg.norm(winner - np.array([winner[0], idx]))
            tmp = np.exp(-(distance ** 2)/self.sdp ) * self.learning_rate
            # print(idx, tmp)
            if tmp < self.threshold:
                # input("press enter")
                j_start = idx
                break
        for idx in range(winner[1], self.j_len):
            distance = np.linalg.norm(winner - np.array([winner[0], idx]))
            tmp = np.exp(-(distance ** 2)/self.sdp ) * self.learning_rate
            # print(idx, tmp)
            if tmp < self.threshold:
                # input("press enter")
                j_end = idx
                break

        print("update", i_start, i_end, j_start, j_end)
        for i in range(i_start, i_end + 1):
            for j in range(j_start, j_end + 1):
                distance = np.linalg.norm(winner - np.array([i, j]))
                # print(self.net[winner[0]][winner[1]] - self.net[i][j], ", ", np.exp(-(distance ** 2)/self.sdp ), ", ", point)
                self.net[i][j] = self.net[i][j] + self.learning_rate * np.exp(-(distance ** 2)/self.sdp ) * (point - self.net[i][j])
        # print("end: ", self.net)
        # input("press enter")
    def training(self, data, drawf):
        winner = None
        dis = 0
        counter = 0
        lock = [False]
        while counter < self.training_times:
            for p in data:
                winner = self.whowin(p)
                self.net_update(p, winner)
                if not lock[0]:
                    lock[0] = True
                    draw_thread = threading.Thread(target=drawf(self.net, lock))
                    draw_thread.start()
                    draw_thread.join()
            counter += 1
    def get_net(self):
        return self.net
    def printnet(self):
        for i in range(self.i_len):
            for j in range(self.j_len):
                print("(" + self.net[i][j][0].__str__() + ", " + self.net[i][j][1].__str__() + ")")
        # for dim in self.dimension:
