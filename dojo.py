from td_net import TD_Net, TD_Net_Wrapper
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
        state_loader = DataLoader(result_states, batch_size=1)
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


if __name__ == '__main__':

    # init model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))
    model = TD_Net_Wrapper(device=device)
    print("Model: {}".format(model))

    n_games = 1000

    # SARSA alg percent of time player explores vs picking optimal move
    epsilon = 0.1

    lambda_param = 0.7
    lr = 0.1

    # modify these
    is_new_output_path = True
    #output_path = 'data/training_results.csv'
    output_path = 'data/training_results.csv'
    game_log_format_str = 'data/self_train_game_logs/game_{}_log.csv'

    is_new_model_path = False
    model_path_format = 'tournament_train_models/td_net_{}_games.pt'
    n_games_trained = 108961
    # if is_new_model_path:
    #     torch.save(model, model_path_format.format(n_games_trained))
    # else:
    #     model.net = torch.load(
    #         model_path_format.format(n_games_trained)).to(device)
    model.net = torch.load(
        model_path_format.format(n_games_trained)).to(device)

    model_path_format = 'self_train_models/td_net_{}_games.pt'
    n_games_trained = 0
    torch.save(model, model_path_format.format(n_games_trained))

    if is_new_output_path:
        train_log = open(output_path, 'w')
        train_log.write("game,avg_loss,final_state_loss,plys\n")
    else:
        train_log = open(output_path, 'a')

    save_iters = 10

    for i in range(n_games):
        game_log = open(game_log_format_str.format(i+n_games_trained+1), 'w')
        game_log.write('player_color,move\n')
        game = Backgammon(Player('p1', 'white'), Player('p2', 'black'))
        game.start_game()
        s = game.encoded_state()
        num_moves = 0
        game_loss = 0
        plys = 0
        optimizer = torch.optim.SGD(model.net.parameters(), lr=lr)
        grads = [[torch.zeros((80, 198)), torch.zeros(80), torch.zeros((4, 80)), torch.zeros(4)]
                 for i in range(4)]
        # display_board(game, game.current_board())
        while(not game.game_is_over()):
            optimizer.zero_grad()
            print("ply: " + str(plys))
            plys += 1
            game.roll_and_register()
            explore = (np.random.rand() < epsilon)
            if explore:
                legal_moves = game.valid_actions()
                move = legal_moves[np.random.choice(
                    range(len(legal_moves)))]
            else:
                move = best_action(game, s, model)
            # display_board(game, game.current_board())
            # out = "Move: "
            # for m in move:
            # out += "({},{}) ".format(m.source(), m.destination())
            # print(out)
            game_copy = game.copy()
            game_copy.make_turn_ignore_legality(move)
            s_1 = game_copy.encoded_state()
            # print(s)
            # print(s_1)
            if (game_copy.is_terminal_state(s_1)):
                final_loss = model.train_final_example(torch.tensor(
                    s, requires_grad=True).reshape(-1).type(torch.FloatTensor), torch.tensor(game_copy.terminal_value(s_1)).type(torch.FloatTensor), optimizer, grads)
                game_loss += final_loss
            else:
                loss, grads = model.train_single_example(torch.tensor(
                    s, requires_grad=True).reshape(-1).type(torch.FloatTensor), torch.tensor(s_1).reshape(-1).type(torch.FloatTensor), optimizer, grads)
                game_loss += loss
            num_moves += 1

            # print(game_loss)
            # print("{},{}\n".format(game.current_player().color, game.move_to_str(move)))
            game_log.write("{},{}\n".format(
                game.current_player().color, game.move_to_str(move)))
            game.make_turn(move)
            s = game.encoded_state()

        train_log.write('{},{},{},{}\n'.format(i + n_games_trained,
                                               game_loss/num_moves, final_loss, plys))
        # state_loader = DataLoader(
        #     [torch.tensor(game.encoded_state()).reshape(-1).type(torch.FloatTensor)], batch_size=4)
        game_log.close()
        # print(model.get_predicted_move_vals(state_loader))
        # print(game.terminal_value(s))
        if (i + 1) % save_iters == 0:
            torch.save(model, model_path_format.format(
                n_games_trained + i + 1))
            train_log.close()
            train_log = open(output_path, 'a')

    train_log.close()

    # state_loader = DataLoader(
    #     [torch.tensor(game.encoded_state()).reshape(-1).type(torch.FloatTensor)], batch_size=4)
    # print(model.get_predicted_move_vals(state_loader))
