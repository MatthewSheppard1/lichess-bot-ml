#referebce: https://suragnair.github.io/posts/alphazero.html

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.BatchNorm2d(256))

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = output + x
        return nn.ReLU(output)

class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        # value head
        self.value_layer = nn.Sequential(nn.Conv2d(256, 1, 1, stride=1, padding=1), nn.BatchNorm2d(1), nn.ReLU(), nn.Flatten(), nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 1), nn.Tanh())

        #policy head
        self.policy_layer = nn.Sequential(nn.Conv2d(256, 128, 1, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Flatten(), nn.Linear(128*8*8, 8*8*64 + 4))

    def forward(self, x):
        value = self.value_layer(x)
        policy = self.policy_layer(x)

        return value, policy


class ChessNN(nn.Module):
    def __init__(self):
        super(ChessNN, self).__init__()
        self.SAVED_WEIGHTS_PATH = "chess-weights.pt"
        self.conv1 = nn.Sequential(nn.Conv2d(12, 256, 3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU())

        # 256x8x8
        self.res_blocks = nn.ParameterList()
        for i in range(19):
            self.res_blocks.append(ResBlock())

        self.out_block = OutBlock()

    def forward(self, x):
        out = self.conv1(x)
        for res_block in self.res_blocks:
            out = res_block(out)
        return self.out_block(out)


    def save_weights(self):
        #should happen at end of game
        torch.save(self.state_dict(), self.SAVED_WEIGHTS_PATH)

    def load_weights(self):
        self.load_state_dict(torch.load(self.SAVED_WEIGHTS_PATH, weights_only=True))


def policy_value_loss(policy_pred, policy, val_pred, val):
    val_loss = F.mse_loss(val_pred, val)
    pol_loss = F.cross_entropy(policy_pred, policy)
    return val_loss + pol_loss


#TODO: do train loop once I know what the data will look like
def train():
    pass