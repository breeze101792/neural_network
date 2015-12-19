import numpy as np
from ploting import paper
import threading
import random
from multiprocessing import Pool, Array, Queue, Process
# for 2 dimension

class SOM:
    def __init__(self, i_len = 100, j_len = 100, sd = 1, learning_rate = 0.4, training_times = 5, status_flag = []):
        self.net = None
        self.dimension = 2
        self.i_len = i_len
        self.j_len = j_len
        self.learning_rate = learning_rate
        self.sdp = (sd ** 2) * 2
        self.training_times = training_times
        self.threshold = 0.001
        self.core_number = 8
        self.tone = 1000
        self.ttwo = 1000
        self.status_flag = status_flag
        print(i_len, j_len, sd, learning_rate, training_times)
    def net_init(self):
        if self.i_len == 1:
            self.net = [[np.array((0, j / (self.j_len - 1)))for j in range(self.j_len)]]
            print(self.net)
            return

        self.net = [[] for _ in range(self.i_len)]
        for i in range(self.i_len):
            # self.net[i] = [np.array((random.random(), random.random()))for j in range(self.j_len)]
            self.net[i] = [np.array((i / (self.i_len - 1), j / (self.j_len - 1)))for j in range(self.j_len)]
    def __whowin_dq(self, point, i_start, i_end, que):
        winner = np.array([0,0])
        winner_ip = 100
        cur_ip = 0
        for i in range(i_start, i_end):
            for j in range(self.j_len):
                cur_ip = np.linalg.norm((point - (self.net[i][j])) if np.sum(self.net[i][j]) != 0 else point)
                if winner_ip > cur_ip:
                    winner_ip = cur_ip
                    winner[0] = i
                    winner[1] = j
        que.put(winner)

    def whowin(self, point):
        cur_ip = 0
        core_number = self.core_number if self.i_len > self.core_number else self.i_len
        # buf = [np.array([-1,-1]) for _ in range(core_number)]
        que = Queue()
        disp = int(self.i_len / core_number)
        counter = 0

        # processes = [Process(target=self.__whowin_dq, args=(point, counter, counter + disp, idx, que)) for idx in range(core_number)]
        processes = []
        for idx in range(core_number):
            if idx == core_number - 1:
                processes.append(Process(target=self.__whowin_dq, args = (point, counter, self.i_len, que)))
            else:
                processes.append(Process(target=self.__whowin_dq, args = (point, counter, counter + disp, que)))
            counter += disp
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        buf = [que.get() for p in processes]
        winner = buf[0]
        winner_ip = np.linalg.norm((point - (self.net[buf[0][0]][buf[0][1]])) if np.sum(self.net[buf[0][0]][buf[0][1]]) != 0 else point)
        for idx in range(1, core_number):
            cur_ip = np.linalg.norm((point - (self.net[buf[idx][0]][buf[idx][1]])) if np.sum(self.net[buf[idx][0]][buf[idx][1]]) != 0 else point)
            if winner_ip > cur_ip:
                winner_ip = cur_ip
                winner = buf[idx]
        bwinner = winner
        #
        # print(buf)
        # print(winner)
        #
        # winner = np.array([0,0])
        # winner_ip = 100
        # for i in range(self.i_len):
        #     for j in range(self.j_len):
        #         cur_ip = np.linalg.norm((point - (self.net[i][j])) if np.sum(self.net[i][j]) != 0 else point)
        #         if winner_ip > cur_ip:
        #             winner_ip = cur_ip
        #             winner[0] = i
        #             winner[1] = j
        # print("winner", winner[0], ", ",winner[1])
        # if np.linalg.norm(winner - bwinner) != 0:
        #     print(buf)
        #     print(winner, bwinner)
        #     input("press enter")
        return winner
        # return np.array([0 ,0])
    def net_update(self, point, winner, times = 0):
        i_start = j_start = 0
        i_end = self.i_len - 1
        j_end = self.j_len - 1
        sd = self.sdp * np.exp(-times/self.tone)
        lr = self.learning_rate * np.exp(-times/self.ttwo)
        tmp = 0
        # print(np.exp(-(0.01)/sd ) * lr, times)
        if np.exp(-(0.1)/sd ) * lr < self.threshold:
            return -1
        # print(winner, point)
        for idx in range(winner[0], -1, -1):
            distance = np.linalg.norm(winner - np.array([idx, winner[1]]))
            tmp = np.exp(-(distance ** 2)/sd ) * lr
            # print(idx, tmp)
            if tmp < self.threshold:
                # input("press enter")
                i_start= idx
                break

        for idx in range(winner[0], self.i_len):
            distance = np.linalg.norm(winner - np.array([idx, winner[1]]))
            tmp = np.exp(-(distance ** 2)/sd ) * lr
            # print(idx, tmp)
            if tmp < self.threshold:
                # input("press enter")
                i_end = idx
                break
        for idx in range(winner[1], -1, -1):
            distance = np.linalg.norm(winner - np.array([winner[0], idx]))
            tmp = np.exp(-(distance ** 2)/sd ) * lr
            # print(idx, tmp)
            if tmp < self.threshold:
                # input("press enter")
                j_start = idx
                break
        for idx in range(winner[1], self.j_len):
            distance = np.linalg.norm(winner - np.array([winner[0], idx]))
            tmp = np.exp(-(distance ** 2)/sd ) * lr
            # print(idx, tmp)
            if tmp < self.threshold:
                # input("press enter")
                j_end = idx
                break

        # print("update", i_start, i_end, j_start, j_end)
        for i in range(i_start, i_end + 1):
            for j in range(j_start, j_end + 1):
                distance = np.linalg.norm(winner - np.array([i, j]))
                # print(self.net[winner[0]][winner[1]] - self.net[i][j], ", ", np.exp(-(distance ** 2)/sd ), ", ", point)
                self.net[i][j] = self.net[i][j] + lr * np.exp(-(distance ** 2)/sd ) * (point - self.net[i][j])
        # print("end: ", self.net)
        # input("press enter")
        return True
    def training(self, data, drawf):
        winner = None
        dis = 0
        counter = 0
        lock = [False]
        while counter < self.training_times:
            n = counter * len(data)
            for idx, p in enumerate(data):
                winner = self.whowin(p)
                if not self.net_update(p, winner, n + idx):
                    return
                if not lock[0]:
                    lock[0] = True
                    draw_thread = threading.Thread(target=drawf(self.net, lock))
                    draw_thread.start()

            counter += 1
    def get_net(self):
        return self.net
    def printnet(self):
        for i in range(self.i_len):
            for j in range(self.j_len):
                print("(" + self.net[i][j][0].__str__() + ", " + self.net[i][j][1].__str__() + ")")
        # for dim in self.dimension:
