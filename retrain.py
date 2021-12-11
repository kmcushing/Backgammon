from td_net import TD_Net
import numpy as np
import torch
from torch.utils.data import DataLoader
from backgammon import Backgammon, Player, Move
import re

def display_board(game, board):
    point_row = ""
    point_label_row = ""
    current_player = game.current_player()
    current_player_color = current_player.color
    next_player_color = game.next_player().color
    current_player_started = game._game_state.starting_player_id() == current_player.id
    board_range = list(range(board.board_size()))
    if not current_player_started:
        board_range.reverse()
    for i in board_range:
        current_player_checkers = board.num_checkers_at_index(
            current_player_color, i)
        next_player_checkers = board.num_checkers_at_index(
            next_player_color, i)
        point_string = f" <{current_player_checkers} | {next_player_checkers}> "
        point_row += point_string
    color_key = f"<{current_player_color} | {next_player_color}>"
    point_label_row = ""
    for i in range(board.board_size()):
        point_label_row += f"{i+1}".center(9)
    print(point_row)
    print(point_label_row)
    print(color_key)
    curr_player = game.current_player()
    opp_player = game._game_state.next_player()
    n_bar_curr = game.current_board().num_checkers_on_bar(curr_player.color)
    n_bar_opp = game.current_board().num_checkers_on_bar(opp_player.color)
    print(
        f"Bar: {curr_player.color}: {n_bar_curr}, {opp_player.color}: {n_bar_opp}")
    print(f"current player direction: <<")

def parse_move(game, ply_str):
    ply_str.strip('\n')
    ply_str = ply_str.split(',')[1]
    move_tokens = re.split("/", ply_str)
    moves = []
    try:
        for token in move_tokens:
            # print(token)
            stripped_token = token.strip(")(")

            value_pair = stripped_token.split('-')
            # print(value_pair)
            move = Move(game.current_player().id,
                        int(value_pair[0]), int(value_pair[1]))
            moves.append(move)
    except:
        moves.append(Move(game.current_player().id, -1, -1))
    return moves

# init model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: {}'.format(device))
model = TD_Net(device=device)
print("Model: {}".format(model))

n_games_logged = 724
n_games_sim = 10000

# modify these
is_new_output_path = True
output_path = 'data/retraining_results.csv'
game_log_format_str = 'data/game_logs/game_{}_log.csv'

is_new_model_path = True
model_path_format = 'retrain_models/td_net_{}_games.pt'
n_games_trained = 0
if is_new_model_path:
    torch.save(model, model_path_format.format(n_games_trained))
else:
    model = torch.load(model_path_format.format(n_games_trained)).to(device)


if is_new_output_path:
    train_log = open(output_path, 'w')
    train_log.write("game,avg_loss,final_state_loss\n")
else:
    train_log = open(output_path, 'a')

save_iters = 10

for i in range(n_games_sim):
    game_to_sim = np.random.randint(0, n_games_logged)
    game_log = open(game_log_format_str.format(game_to_sim, 'r'))
    game = Backgammon(Player('p1', 'white'), Player('p2', 'black'))
    game.start_game()
    game_loss = 0
    num_moves = 0
    s = game.encoded_state()
    # display_board(game, game.current_board())
    model.zero_grad()
    for ply in game_log.readlines()[1:]:
        # register roll based off move
        # print(num_moves)
        game.roll_and_register()
        move = parse_move(game, ply)
        
        # display_board(game, game.current_board())
        # out = "Move: "
        # for m in move:
        # out += "({},{}) ".format(m.source(), m.destination())
        # print(out)

        s_1 = game.simulate_single_move(s, move)
        # print(s)
        # print(s_1)
        if (game.is_terminal_state(s_1)):
            final_loss = model.train_final_example(torch.tensor(
                s).reshape(-1).type(torch.FloatTensor), torch.tensor(game.terminal_value(s_1)).type(torch.FloatTensor))
            game_loss += final_loss
        else:
            game_loss += model.train_single_example(torch.tensor(
                s).reshape(-1).type(torch.FloatTensor), torch.tensor(s_1).reshape(-1).type(torch.FloatTensor))

        num_moves += 1
        # print(game_loss)
        # print("{},{}\n".format(game.current_player().color, game.move_to_str(move)))
        game.make_turn_ignore_legality(move)
        s = game.encoded_state()

    train_log.write('{},{},{}\n'.format(i + n_games_trained,
                                           game_loss/num_moves, final_loss))
    state_loader = DataLoader(
        [torch.tensor(game.encoded_state()).reshape(-1).type(torch.FloatTensor)], batch_size=4)
    game_log.close()
    # print(model.get_predicted_move_vals(state_loader))
    # print(game.terminal_value(s))
    if (i + 1) % save_iters == 0:
        torch.save(model, model_path_format.format(n_games_trained + i + 1))
        train_log.close()
        train_log = open(output_path, 'a')

train_log.close()

# state_loader = DataLoader(
#     [torch.tensor(game.encoded_state()).reshape(-1).type(torch.FloatTensor)], batch_size=4)
# print(model.get_predicted_move_vals(state_loader))
