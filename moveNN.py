#referebce: https://suragnair.github.io/posts/alphazero.html

import torch
import torch.nn as nn
import torch.nn.functional as F

class MoveNN(nn.Module):
    def __init__(self):
        super(MoveNN, self).__init__()
        self.fc1 = nn.Linear(34, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 30)
    def update_stored_weights(self, tensors: list[list[torch.tensor]]):
        #should happen at end of game
        f = open(self.SAVED_STATES_PATH, "w")
        for network in tensors:
            for t in network:
                temp = ""
                for i in t:
                    temp += str(i) + ","
                temp = temp[:-1] + "\n"
                f.write(temp)
        f.close()

    def read_stored_weights(self) -> list[list[torch.tensor]]:
        #should happen at initialization of engine or beginning of game
        f = open(self.SAVED_STATES_PATH, "r")
        tensors = [[]]
        for line in f:
            curr_tensor = 0
            if(line == "\n"):
                curr_tensor += 1
                tensors.append([])
            temp = line.split(",")
            for i in range(len(temp)):
                temp[i] = float(temp[i])
            tensors[curr_tensor].append(temp)
        f.close()
        return tensors
    def set_weights(self, weights: list[torch.tensor]):
        self.fc1.weight.data = weights[0]
        self.fc2.weight.data = weights[1]
        self.fc3.weight.data = weights[2]

    def forward(self, input):
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        output = self.fc3(f2)