import random

import chess
import numpy as np
import math
from encoder_decoder import *
import torch.nn.functional as F
import torch

#TODO: Implent self-play function
#self play function that returns dataset

class MCTSNode:
    def __init__(self, board: chess.Board, parent):
        self.board = board
        self.parent = parent
        self.moves = board.legal_moves
        self.terminal = board.outcome() is not None
        self.children = dict()
        for m in self.moves:
            self.children[m] = None

        # Set up stats for MCTS
        # Number of visits to this node
        self.n = 0

        # Total number of wins from this node (win = +1, loss = -1, tie = +0)
        # Note: these wins are from the perspective of the PARENT node of this node
        #       So, if self.player_number wins, that is -1, while if self.other_player_number wins
        #       that is a +1.  (Since parent will be using our UCB value to make choice)
        self.w = 0

        # c value to be used in the UCB calculation
        self.c = np.sqrt(2)

    def print_tree(self):
        # Debugging utility that will print the whole subtree starting at this node
        print("****")
        self.print_node()
        for m in self.moves:
            if self.children[m]:
                self.children[m].print_tree()
        print("****")

    def print_node(self):
        # Debugging utility that will print this node's information
        print('Total Node visits and wins: ', self.n, self.w)
        print('Children: ')
        for m in self.moves:
            if self.children[m] is None:
                print('   ', m, ' is None')
            else:
                print('   ', m, ':', self.children[m].n, self.children[m].w, 'UB: ',
                      self.children[m].upper_bound(self.n))

    def max_child(self):
        # Return the most visited child
        # This is used at the root node to make a final decision
        max_n = 0
        max_move = None

        for m in self.moves:
            if(self.children[m] is not None):
                if self.children[m].n > max_n:
                    max_n = self.children[m].n
                    max_move = m
        return max_move

    def upper_bound(self, N):
        # This function returns the UCB for this node
        # N is the number of samples for the parent node, to be used in UCB calculation

        # To do: return the UCB for this node (look in __init__ to see the values you can use)

        return (self.w / self.n) + math.sqrt(2) * math.sqrt(math.log(N) / self.n)

    def select(self):
        # This recursive function combines the selection and expansion steps of the MCTS algorithm
        # It will return either:
        # A terminal node, if this is the node selected
        # The new node added to the tree, if a leaf node is selected

        max_ub = -np.inf  # Track the best upper bound found so far
        max_child = None  # Track the best child found so far

        if self.terminal:
            # If this is a terminal node, then return it (the game is over)
            return self

        # For all of the children of this node
        for m in self.moves:
            if self.children[m] is None:
                # If this child doesn't exist, then create it and return it
                new_board = self.board.copy()  # Copy board/state for the new child
                new_board.push(m)  # Make the move in the state

                self.children[m] = MCTSNode(new_board, self)  # Create the child node
                return self.children[m]  # Return it

            # Child already exists, get it's UCB value
            current_ub = self.children[m].upper_bound(self.n)

            # Compare to previous best UCB
            if current_ub > max_ub:
                max_ub = current_ub
                max_child = m

        # Recursively return the select result for the best child
        return self.children[max_child].select()

    def simulate(self, network):
        pass
        # This function will simulate a random game from this node's state and then call back on its
        # parent with the result

        # YOUR MCTS TASK 2 CODE GOES HERE

        # Pseudocode in comments:
        #################################
        # If this state is terminal (meaning the game is over) AND it is a winning state for self.other_player_number
        #   Then we are done and the result is 1 (since this is from parent's perspective)
        #
        # Else-if this state is terminal AND is a winning state for self.player_number
        #   Then we are done and the result is -1 (since this is from parent's perspective)
        #
        # Else-if this is not a terminal state (if it is terminal and a tie (no-one won, then result is 0))
        #   Then we need to perform the random rollout
        #      1. Make a copy of the board to modify
        #      2. Keep track of which player's turn it is (first turn is current nodes self.player_number)
        #      3. Until the game is over:
        #            3.1  Make a random move for the player who's turn it is
        #            3.2  Check to see if someone won or the game ended in a tie
        #                 (Hint: you can check for a tie if there are no more valid moves)
        #            3.3  If the game is over, store the result
        #            3.4  If game is not over, change the player and continue the loop
        #
        # Update this node's total reward (self.w) and visit count (self.n) values to reflect this visit and result

        if (self.terminal and self.board.outcome().winner == self.board.turn):
            result = -1
        elif (self.terminal and self.board.outcome().winner != self.board.turn):
            result = 1
        elif (self.terminal):
            result = 0
        else:
            with torch.no_grad():
                result, policy = network(encode_board(self.board).unsqueeze(0))

        self.back(result)
        # Back-propagate this result
        # You do this by calling back on the parent of this node with the result of this simulation
        #    This should look like: self.parent.back(result)
        # Tip: you need to negate the result to account for the fact that the other player
        #    is the actor in the parent node, and so the scores will be from the opposite perspective

    def back(self, score):
        # This updates the stats for this node, then backpropagates things
        # to the parent (note the inverted score)
        self.n += 1
        self.w += score
        if self.parent is not None:
            self.parent.back(-score)  # Score inverted before passing along
    def delete_all(self):
        for m in self.moves:
            if(self.children[m] is not None):
                self.children[m].delete_all()
                del self.children[m]



def get_mcts_move(network, root, max_iterations):




    # Run our MCTS iterations
    for i in range(max_iterations):
        # Select + Expand
        cur_node = root.select()

        # Simulate + backpropate
        cur_node.simulate(network)
    move = root.max_child()


    return move



def self_play(network, num_games, max_mcts_iter, game_depth_limit):
    data = []
    for i in range(num_games):
        states = []
        moves = []
        masks = []
        board = chess.Board()
        count = 0
        root_node = MCTSNode(board, None)
        while(board.outcome() is None):
            right_move = get_mcts_move(network, root_node, max_mcts_iter)

            states.append(encode_board(board))
            with torch.no_grad():
                result, policy = network(encode_board(board).unsqueeze(0))
            mask = make_move_mask(board)
            masks.append(mask)
            policy = policy.squeeze(0) * mask
            move_list, probs = decode_mask(policy, board)
            for j in range(len(move_list)):
                if(move_list[j] == right_move):
                    break
            moves.append(j)
            probs = F.softmax(probs, 0)

            sample_sum = 0
            rand = random.random()
            idx = 0
            for j in range(len(probs)):
                sample_sum += probs[j]
                if(sample_sum >= rand):
                    idx = j
                    break


            for m in root_node.moves:
                if (m != move_list[idx]):
                    if(root_node.children[m] is not None):
                        root_node.children[m].delete_all()

            root_node = root_node.children[move_list[idx]]
            board.push(move_list[idx])
            if(count >= game_depth_limit):
                break
            count += 1
            print(count)
        result = board.outcome(claim_draw=True)
        winner = None
        if(result is not None):
            winner = result.winner

        print("Game Over")

        #Add game data to data list
        for j in range(len(states)):
            z = 1
            if(winner == None):
                z = 0
            elif(states[j].turn != winner):
                z = -1

            data.append((states[j], moves[j], z, masks[j]))
    print("Self-Play Complete")
    return data










