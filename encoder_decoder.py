import chess
import numpy as np

# Layers:
# 0-11 piece representation

# Mask Layers
# 0-27 all dists and dir rook moves
# 28-55 all dists and dir bishop moves
# 56-63 knight moves

#Encode the board from chess.Board representation to NN input representation
def encode_board(board: chess.Board):
    encoded = np.zeros([12, 8, 8])

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if(piece != None):
            i = piece.piece_type - 1
            if(piece.color == chess.BLACK):
                i += 6

            encoded[i, int(square / 8), square % 8] = 1

    return encoded


#TODO: idk if i'll actually need this, so i'll finish if I find a need for it
def decode_board(encoded) -> chess.Board:
    decoded = np.zeros([64])

    for i in range(len(encoded)):
        for j in range(len(encoded[0])):
            for k in range(len(encoded[0][0])):
                if(k == 1):
                    pass
    return chess.Board()


def get_move_mask_index(dist, direction):
    #directions: 0-7 clockwise starting up
    #0-55
    return direction * 7 + dist - 1


def make_move_mask(board: chess.Board):
    mask = np.zeros([8, 8, 64]).astype(int)

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
    return mask


def decode_mask(mask) -> list[(chess.Move, float)]:
    moves = []

    vals = "abcdefgh"
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            for k in range(len(mask[0][0])):
                coords = [i, j + 1, 0, 0]
                if(mask[i][j][k] > 0):
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
                    moves.append((chess.Move.from_uci(move_uci), mask[i][j][k]))

    return moves
