import chess
import torch
import numpy as np

# Layers:
# 0-11 piece representation

# Mask Layers
# 0-27 all dists and dir rook moves
# 28-55 all dists and dir bishop moves
# 56-63 knight moves

#Encode the board from chess.Board representation to NN input representation
def encode_board(board: chess.Board):
    encoded = torch.zeros([12, 8, 8])

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if(piece != None):
            i = piece.piece_type - 1
            if(piece.color == chess.BLACK):
                i += 6

            encoded[i, int(square / 8), square % 8] = 1

    return encoded


#TODO: idk if i'll actually need this, so i'll finish if I find a need for it
def decode_board(encoded, batch_size):
    boards = []
    key = ['P', 'N', 'B', 'R', 'Q', 'K']

    for b in range(batch_size):
        decoded = []
        for i in range(8):
            decoded.append([])
            for j in range(8):
                decoded[i].append(" ")

        for k in range(len(encoded[0])):
            for i in range(len(encoded[0][0])):
                for j in range(len(encoded[0][0][0])):
                    if(encoded[b][k][i][j] > 0):
                        c = key[k % 6]
                        if(k >= 6):
                            c = c.lower()
                        decoded[i][j] = c
        s = ""
        space_count = 0
        for i in range(8):
            for j in range(8):
                if(decoded[i][j] == " "):
                    space_count += 1
                elif(space_count > 0):
                    s += str(space_count) + decoded[i][j]
                    space_count = 0
                else:
                    s += decoded[i][j]
            if (space_count > 0):
                s += str(space_count)
                space_count = 0
            if(i < 7):
                s += "/"

        s += " w"

        boards.append(chess.Board(s))

    return boards


def get_move_mask_index(dist, direction):
    #directions: 0-7 clockwise starting up
    #0-55
    return direction * 7 + dist - 1

def get_move_index(move: chess.Move, board: chess.Board):
    piece = board.piece_at(move.from_square)
    vals = "abcdefgh"
    move_str = move.uci()
    coords = [int(move_str[1]) - 1, int(move_str[3]) - 1, vals.find(move_str[0]), vals.find(move_str[2])]
    if (piece.symbol().lower() == "n"):
        direction = 0
        if (coords[0] - coords[1] == -2 and coords[2] - coords[3] == -1):
            direction = 0
        elif (coords[0] - coords[1] == -1 and coords[2] - coords[3] == -2):
            direction = 1
        elif (coords[0] - coords[1] == 1 and coords[2] - coords[3] == -2):
            direction = 2
        elif (coords[0] - coords[1] == 2 and coords[2] - coords[3] == -1):
            direction = 3
        elif (coords[0] - coords[1] == 2 and coords[2] - coords[3] == 1):
            direction = 4
        elif (coords[0] - coords[1] == 1 and coords[2] - coords[3] == 2):
            direction = 5
        elif (coords[0] - coords[1] == -1 and coords[2] - coords[3] == 2):
            direction = 6
        elif (coords[0] - coords[1] == -2 and coords[2] - coords[3] == 1):
            direction = 7
        return coords[2] * 64 * 8 + coords[0] * 64 + 56 + direction
    else:
        dist = max(abs(coords[0] - coords[1]), abs(coords[2] - coords[3]))
        direction = 0
        if (coords[0] - coords[1] < 0 and coords[2] - coords[3] < 0):
            direction = 1
        elif (coords[0] - coords[1] > 0 and coords[2] - coords[3] < 0):
            direction = 3
        elif (coords[0] - coords[1] > 0 and coords[2] - coords[3] > 0):
            direction = 5
        elif (coords[0] - coords[1] < 0 and coords[2] - coords[3] > 0):
            direction = 7
        elif (coords[0] - coords[1] < 0 and coords[2] - coords[3] == 0):
            direction = 0
        elif (coords[0] - coords[1] == 0 and coords[2] - coords[3] < 0):
            direction = 2
        elif (coords[0] - coords[1] > 0 and coords[2] - coords[3] == 0):
            direction = 4
        elif (coords[0] - coords[1] == 0 and coords[2] - coords[3] > 0):
            direction = 6

        return coords[2] * 64 * 8 + coords[0] * 64 + get_move_mask_index(dist, direction)


def make_move_mask(board: chess.Board):
    mask = torch.zeros([8, 8, 64])

    vals = "abcdefgh"
    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        move_str = move.uci()
        coords = [int(move_str[1]) - 1, int(move_str[3]) - 1, vals.find(move_str[0]), vals.find(move_str[2])]
        if(piece.symbol().lower() == "n"):
            direction = 0
            if (coords[0] - coords[1] == -2 and coords[2] - coords[3] == -1):
                direction = 0
            elif (coords[0] - coords[1] == -1 and coords[2] - coords[3] == -2):
                direction = 1
            elif (coords[0] - coords[1] == 1 and coords[2] - coords[3] == -2):
                direction = 2
            elif (coords[0] - coords[1] == 2 and coords[2] - coords[3] == -1):
                direction = 3
            elif (coords[0] - coords[1] == 2 and coords[2] - coords[3] == 1):
                direction = 4
            elif (coords[0] - coords[1] == 1 and coords[2] - coords[3] == 2):
                direction = 5
            elif (coords[0] - coords[1] == -1 and coords[2] - coords[3] == 2):
                direction = 6
            elif (coords[0] - coords[1] == -2 and coords[2] - coords[3] == 1):
                direction = 7
            mask[coords[2], coords[0], 56 + direction] = 1
        else:
            dist = max(abs(coords[0] - coords[1]), abs(coords[2] - coords[3]))
            direction = 0
            if(coords[0] - coords[1] < 0 and coords[2] - coords[3] < 0):
                direction = 1
            elif(coords[0] - coords[1] > 0 and coords[2] - coords[3] < 0):
                direction = 3
            elif(coords[0] - coords[1] > 0 and coords[2] - coords[3] > 0):
                direction = 5
            elif(coords[0] - coords[1] < 0 and coords[2] - coords[3] > 0):
                direction = 7
            elif(coords[0] - coords[1] < 0 and coords[2] - coords[3] == 0):
                direction = 0
            elif(coords[0] - coords[1] == 0 and coords[2] - coords[3] < 0):
                direction = 2
            elif(coords[0] - coords[1] > 0 and coords[2] - coords[3] == 0):
                direction = 4
            elif(coords[0] - coords[1] == 0 and coords[2] - coords[3] > 0):
                direction = 6

            mask[coords[2], coords[0], get_move_mask_index(dist, direction)] = 1
    return mask.view([-1])

def encode_move(move: chess.Move):
    move_t = torch.zeros([8, 8])
    move_t[int(move.from_square / 8)][move.from_square % 8] = 1
    move_t[int(move.to_square / 8)][move.to_square % 8] = 1

    return move_t
def decode_move(move_t):
    vals = "abcdefgh"
    coords = [0, 0, 0, 0]
    for i in range(len(move_t)):
        for j in range(len(move_t[0])):
            if(move_t[i][j] == 1):
                coords[0] = i
                coords[1] = j + 1
            if (move_t[i][j] == 2):
                coords[2] = i
                coords[3] = j + 1
    return chess.Move.from_uci(vals[coords[0]] + str(coords[1]) + vals[coords[2]] + str(coords[3]))




def decode_mask(mask, boards: list[chess.Board], batch_size=None):
    if(batch_size is None):
        moves = []
        probs = []
        mask = mask.view([8, 8, 64])
        vals = "abcdefgh"
        for i in range(len(mask)):
            for j in range(len(mask[0])):
                for k in range(len(mask[0][0])):
                    coords = [i, j + 1, 0, 0]
                    if(mask[i][j][k] != 0):
                        if(k > 55):
                            direction = k - 56

                            if (direction == 0):
                                coords[2] = coords[0] + 1
                                coords[3] = coords[1] + 2
                            elif (direction == 1):
                                coords[2] = coords[0] + 2
                                coords[3] = coords[1] + 1
                            elif (direction == 2):
                                coords[2] = coords[0] + 2
                                coords[3] = coords[1] - 1
                            elif (direction == 3):
                                coords[2] = coords[0] + 1
                                coords[3] = coords[1] - 2
                            elif (direction == 4):
                                coords[2] = coords[0] - 1
                                coords[3] = coords[1] - 2
                            elif (direction == 5):
                                coords[2] = coords[0] - 2
                                coords[3] = coords[1] - 1
                            elif (direction == 6):
                                coords[2] = coords[0] - 2
                                coords[3] = coords[1] + 1
                            elif (direction == 7):
                                coords[2] = coords[0] - 1
                                coords[3] = coords[1] + 2

                        else:
                            direction = int(k / 7)
                            dist = (k % 7) + 1
                            mods = [0, 0]

                            if (direction in [0, 1, 7]):
                                mods[1] = 1
                            if (direction in [1, 2, 3]):
                                mods[0] = 1
                            if (direction in [3, 4, 5]):
                                mods[1] = -1
                            if (direction in [5, 6, 7]):
                                mods[0] = -1
                            coords[2] = coords[0] + (mods[0] * dist)
                            coords[3] = coords[1] + (mods[1] * dist)

                        move_uci = vals[coords[0]] + str(coords[1]) + vals[coords[2]] + str(coords[3])
                        if((coords[3] == 1 or coords[3] == 8) and boards[0].piece_type_at(chess.parse_square(vals[coords[0]] + str(coords[1]))) == 1):
                            move_uci += "q"

                        moves.append(chess.Move.from_uci(move_uci))
                        probs.append(mask[i][j][k])
        probs = torch.tensor(probs)
        return moves, probs
    else:
        mask = mask.view([batch_size, 8, 8, 64])
        vals = "abcdefgh"
        batch_probs = []
        batch_moves = []
        for b in range(batch_size):
            moves = []
            probs = []
            for i in range(len(mask[0])):
                for j in range(len(mask[0][0])):
                    for k in range(len(mask[0][0][0])):
                        coords = [i, j + 1, 0, 0]
                        if (mask[b][i][j][k] != 0):
                            if (k > 55):
                                direction = k - 56

                                if (direction == 0):
                                    coords[2] = coords[0] + 1
                                    coords[3] = coords[1] + 2
                                elif (direction == 1):
                                    coords[2] = coords[0] + 2
                                    coords[3] = coords[1] + 1
                                elif (direction == 2):
                                    coords[2] = coords[0] + 2
                                    coords[3] = coords[1] - 1
                                elif (direction == 3):
                                    coords[2] = coords[0] + 1
                                    coords[3] = coords[1] - 2
                                elif (direction == 4):
                                    coords[2] = coords[0] - 1
                                    coords[3] = coords[1] - 2
                                elif (direction == 5):
                                    coords[2] = coords[0] - 2
                                    coords[3] = coords[1] - 1
                                elif (direction == 6):
                                    coords[2] = coords[0] - 2
                                    coords[3] = coords[1] + 1
                                elif (direction == 7):
                                    coords[2] = coords[0] - 1
                                    coords[3] = coords[1] + 2

                            else:
                                direction = int(k / 7)
                                dist = (k % 7) + 1
                                mods = [0, 0]

                                if (direction in [0, 1, 7]):
                                    mods[1] = 1
                                if (direction in [1, 2, 3]):
                                    mods[0] = 1
                                if (direction in [3, 4, 5]):
                                    mods[1] = -1
                                if (direction in [5, 6, 7]):
                                    mods[0] = -1

                                coords[2] = coords[0] + (mods[0] * dist)
                                coords[3] = coords[1] + (mods[1] * dist)

                            move_uci = vals[coords[0]] + str(coords[1]) + vals[coords[2]] + str(coords[3])
                            if ((coords[3] == 1 or coords[3] == 8) and boards[b].piece_type_at(
                                    chess.parse_square(vals[coords[0]] + str(coords[1]))) == 1):
                                move_uci += "q"

                            moves.append(chess.Move.from_uci(move_uci))
                            probs.append(mask[b][i][j][k])

            probs = torch.tensor(probs)
            batch_probs.append(probs)
            batch_moves.append(moves)
        return batch_moves, batch_probs
