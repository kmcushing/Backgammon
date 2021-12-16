from td_net import TD_Net, TD_Net_Wrapper
import numpy as np
import torch
from torch.utils.data import DataLoader
from backgammon import Backgammon, Player
from dojo import best_action, display_board


# init model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: {}'.format(device))
model1 = TD_Net_Wrapper(device=device)
model2 = TD_Net_Wrapper(device=device)

model_n1 = 0
model_n2 = 108961

n_games = 100

# modify these
is_new_output_path = True
# output_path = 'data/training_results.csv'
output_path = 'data/training_competition_results.csv'
game_log_format_str = ('data/competition_logs/TD' + str(model_n1) + '_vs_TD'
                       + str(model_n2) + '_game_{}_log.csv')

model_path_format = 'tournament_train_models/td_net_{}_games.pt'

model1.net = torch.load(
    model_path_format.format(model_n1), map_location=device)

model2.net = torch.load(
    model_path_format.format(model_n2), map_location=device)


if is_new_output_path:
    train_log = open(output_path, 'w')
    train_log.write("game,final_value,winner,plys\n")
else:
    train_log = open(output_path, 'a')

save_iters = 10

players = [model1, model2]
numbers = [model_n1, model_n2]

games_played = 0

for i in range(n_games):
    game_log = open(game_log_format_str.format(games_played + i), 'w')
    game_log.write('player_color,move\n')
    game = Backgammon(Player('p1', 'white'), Player('p2', 'black'))
    game.start_game_deterministic_first_player()
    s = game.encoded_state()
    num_moves = 0
    game_loss = 0
    plys = 0
    print(f'Game {i}')
    display_board(game, game.current_board())
    start_player_idx = np.random.randint(0, 2)
    cp_idx = start_player_idx
    while(not game.game_is_over()):
        if plys > 0:
            cp_idx = (cp_idx + 1) % 2
        print("ply: " + str(plys))
        plys += 1
        game.roll_and_register()
        move = best_action(game, s, players[cp_idx])
        move_str = 'Move: '
        for m in move:
            move_str += f'{m.source()}-{m.destination()}/'
        print(move_str)
        num_moves += 1
        game_log.write("{},{}\n".format(
            game.current_player().color, game.move_to_str(move)))
        game.make_turn(move)
        display_board(game, game.current_board())
        s = game.encoded_state()

    final_value = game.terminal_value(s)
    train_log.write('{},{},{},{}\n'.format(
        games_played + i, final_value, numbers[cp_idx], plys))
    # state_loader = DataLoader(
    #     [torch.tensor(game.encoded_state()).reshape(-1).type(torch.FloatTensor)], batch_size=4)
    game_log.close()
    # print(model.get_predicted_move_vals(state_loader))
    # print(game.terminal_value(s))
    if (i + 1) % save_iters == 0:
        train_log.close()
        train_log = open(output_path, 'a')

train_log.close()

# state_loader = DataLoader(
#     [torch.tensor(game.encoded_state()).reshape(-1).type(torch.FloatTensor)], batch_size=4)
# print(model.get_predicted_move_vals(state_loader))
