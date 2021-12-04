from td_net import TD_Net
import numpy as np
import torch
from torch.utils.data import DataLoader
from backgammon import Backgammon, Player

def best_action(game, state, model):
    # use mode to compute best move according to learned policy
    with torch.no_grad():
        legal_moves = game.valid_actions()
        result_states = [torch.tensor(game.simulate_single_move(
            state, move)).reshape(-1).type(torch.FloatTensor)
            for move in legal_moves]
        state_loader = DataLoader(result_states, batch_size=4)
        if game.current_player_direction() == 1:
            move = legal_moves[torch.argmax(
                model.get_predicted_move_vals(state_loader))]
        else:
            move = legal_moves[torch.argmin(
                model.get_predicted_move_vals(state_loader))]
    return move


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


# init model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: {}'.format(device))
model = TD_Net(device=device)
print("Model: {}".format(model))

n_games = 1

# SARSA alg
epsilon = 0.01

is_new_output_path = True
output_path = 'data/training_results.csv'

if is_new_output_path:
    f = open(output_path, 'w')
    f.write("game,avg_loss,plys\n")
else:
    f = open(output_path, 'a')

for i in range(n_games):
    game = Backgammon(Player('p1', 'white'), Player('p2', 'black'))
    game.start_game()
    s = game.encoded_state()
    num_moves = 0
    game_loss = 0
    plys = 0
    display_board(game, game.current_board())
    model.zero_grad()
    while(not game.game_is_over()):
        print("ply: " + str(plys))
        plys += 1
        game.roll_and_register()
        explore = (np.random.rand() < epsilon)
        if explore:
            move = game.valid_actions()[np.random.choice(
                range(len(game.valid_actions())))]
            move_weight = 0
        else:
            move = best_action(game, s, model)
            move_weight = 1
        display_board(game, game.current_board())
        out = "Move: "
        for m in move:
            out += "({},{}) ".format(m.source(), m.destination())
        print(out)

        s_1 = game.simulate_single_move(s, move)
        print(s)
        print(s_1)
        if (game.is_terminal_state(s_1)):
            game_loss += move_weight * model.train_final_example(torch.tensor(
                s).reshape(-1).type(torch.FloatTensor), torch.tensor(game.terminal_value(s_1)).type(torch.FloatTensor))
            num_moves += move_weight
        else:
            game_loss += move_weight * model.train_single_example(torch.tensor(
                s).reshape(-1).type(torch.FloatTensor), torch.tensor(s_1).reshape(-1).type(torch.FloatTensor))
            num_moves += move_weight

        print(game_loss)

        game.make_turn(move)
        s = game.encoded_state()

    f.write('{},{},{}\n'.format(i, game_loss/num_moves, plys))
    state_loader = DataLoader(
        [torch.tensor(game.encoded_state()).reshape(-1).type(torch.FloatTensor)], batch_size=4)
    print(model.get_predicted_move_vals(state_loader))
    print(game.terminal_value(s))

f.close()

# state_loader = DataLoader(
#     [torch.tensor(game.encoded_state()).reshape(-1).type(torch.FloatTensor)], batch_size=4)
# print(model.get_predicted_move_vals(state_loader))
