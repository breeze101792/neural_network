import numpy as np
from ploting import paper

# for 2 dimension
class SOM:
    def __init__(self, i_len = 100, j_len = 100, sd = 1):
        self.net = None
        self.dimension = 2
        self.i_len = i_len
        self.j_len = j_len
        self.learning_rate = 0.1
        self.sdp = (sd ** 2) * 2
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
        for i in range(self.i_len):
            for j in range(self.j_len):
                distance = np.linalg.norm(winner - np.array([i, j]))# is wrong here TODO
                # print(self.net[winner[0]][winner[1]] - self.net[i][j], ", ", np.exp(-(distance ** 2)/self.sdp ), ", ", point)
                self.net[i][j] = self.net[i][j] + self.learning_rate * np.exp(-(distance ** 2)/self.sdp ) * (point - self.net[i][j])
    def training(self, data):
        winner = None
        dis = 0
        counter = 0

        while counter < 50:
            for p in data:
                winner = self.whowin(p)
                self.net_update(p, winner)
            counter += 1
    def get_net(self):
        return self.net
    def printnet(self):
        for i in range(self.i_len):
            for j in range(self.j_len):
                print("(" + self.net[i][j][0].__str__() + ", " + self.net[i][j][1].__str__() + ")")
        # for dim in self.dimension:
