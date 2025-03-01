#referebce: https://suragnair.github.io/posts/alphazero.html
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder_decoder import *
from self_play_engine import self_play
from torch.utils.data import Dataset, DataLoader
import multiprocessing


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.BatchNorm2d(256))
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = output + x
        return self.relu(output)

class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        # value head
        self.value_layer = nn.Sequential(nn.Conv2d(256, 1, 1, stride=1, padding=0), nn.BatchNorm2d(1), nn.ReLU(), nn.Flatten(), nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 1), nn.Tanh())

        #policy head
        self.policy_layer = nn.Sequential(nn.Conv2d(256, 128, 1, stride=1, padding=0), nn.BatchNorm2d(128), nn.ReLU(), nn.Flatten(), nn.Linear(128*8*8, 8*8*64))

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


def train(network, optim, dataloader, num_epochs):
    for i in range(num_epochs):
        for x, y, z, mask in dataloader:
            value, policy = network(x)
            policy = policy * mask
            policy = F.normalize(policy, dim=1)
            loss = policy_value_loss(policy, y, value.squeeze(), z)
            loss.backward()
            optim.step()
            optim.zero_grad()
def evaluate(model1, model2, num_games, required_win_percent):
    wins = [0, 0]
    for i in range(num_games):
        board = chess.Board()
        while (board.outcome() is None):
            print(board)
            if((board.turn == chess.WHITE and i < int(num_games / 2)) or (board.turn == chess.BLACK and i >= int(num_games / 2))):
                with torch.no_grad():
                    result, policy = model1(encode_board(board).unsqueeze(0))
            elif((board.turn == chess.WHITE and i >= int(num_games / 2)) or (board.turn == chess.BLACK and i < int(num_games / 2))):
                with torch.no_grad():
                    result, policy = model2(encode_board(board).unsqueeze(0))
            with torch.no_grad():
                policy = policy.squeeze(0) * make_move_mask(board)
                policy = F.normalize(policy, dim=0)
            move_list, probs = decode_mask(policy, [board])

            board.push(move_list[torch.argmax(probs)])

        if((board.outcome().winner == chess.WHITE and i < int(num_games / 2)) or (board.outcome().winner == chess.BLACK and i >= int(num_games / 2))):
            wins[0] += 1
        elif((board.outcome().winner == chess.WHITE and i >= int(num_games / 2)) or (board.outcome().winner == chess.BLACK and i < int(num_games / 2))):
            wins[1] += 1

    if(wins[1] / (wins[0] + 0.00001) > required_win_percent):
        print("Challenger Wins")
        print(wins[1] / (wins[0] + 0.00001))
        return model2
    else:
        print("Old Model Wins")
        print(wins[0] / (wins[1] + 0.00001))
        return model1


class ChessDataSet(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def pipeline(num_processes=5):
    model = ChessNN()
    model.load_weights()

    pool = multiprocessing.Pool(processes=num_processes)
    inputs = []
    for i in range(num_processes):
        inputs.append((model, 20, 50, 150))

    temp_data = pool.starmap(self_play, inputs)
    data = []
    for x in temp_data:
        for y in x:
            data.append(y)

    dataloader = DataLoader(ChessDataSet(data), batch_size=16, shuffle=True)
    optim = torch.optim.Adam(model.parameters())

    train(model, optim, dataloader, 10)
    print("Training Complete")

    old_model = ChessNN()
    old_model.load_weights()

    final_model = evaluate(old_model, model, 10, 0.55)
    final_model.save_weights()
    print("Evaluation Complete")


if __name__ == '__main__':
    pipeline()


# Pipeline
# --------
# Store weights of model
# Self-play
# train on self-play dataset
# evaluate new model against old
# model = best model of the 2
