from td_net import TD_Net
import numpy as np
import torch
from torch.utils.data import DataLoader
# import backgammon game


class FakeGame():
    def __init__(self):
        self.cp = RED
        self.over = False
        self.num_moves = 0

    def reset(self):
        # reset to new game state
        pass

    def valid_actions(self):
        return ['({},{}) ({},{})'.format(i, i + 2, j, j + 4)
                for i in [3, 6, 9] for j in [14, 16, 18]]

    def switch_cp(self):
        self.cp = (self.cp + 1) % 2

    def state(self):
        # can return state tensor in order of current player first
        return np.random.randint(0, 5, (2, 26))

    def move(self, move):
        self.num_moves += 1
        if self.num_moves == 5:
            self.over = True
        self.switch_cp()

    def simulate_single_move(self, move):
        return np.random.randint(0, 5, (2, 26))


def best_action(game, model):
    # use mode to compute best move according to learned policy
    with torch.no_grad():
        legal_moves = game.valid_actions()
        result_states = [torch.tensor(game.simulate_single_move(
            move)).reshape(-1).type(torch.FloatTensor)
            for move in legal_moves]
        state_loader = DataLoader(result_states, batch_size=4)
        move = legal_moves[torch.argmax(
            model.get_predicted_move_vals(state_loader))]
    return move


RED = 0
BLACK = 1

# init model
model = TD_Net()
model.zero_grad()

# init backgammon game
game = FakeGame()

n_games = 2

# SARSA alg
epsilon = 0.01

for i in range(n_games):
    s = game.state()
    while(not game.over):
        rand = (np.random.rand() < epsilon)
        if rand:
            move = np.random.choice(game.valid_actions())
        else:
            move = best_action(game, model)

        print(s)
        print(move)

        s_1 = game.simulate_single_move(move)
        model.train_single_example(torch.tensor(
            s).reshape(-1).type(torch.FloatTensor), torch.tensor(s_1).reshape(-1).type(torch.FloatTensor))

        game.move(move)
        s = s_1

        print(s)
