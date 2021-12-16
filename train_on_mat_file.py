from td_net import TD_Net_Wrapper
from gammon_net import Gammon_Net_Wrapper
import numpy as np
import torch
from torch.utils.data import DataLoader
from backgammon import Backgammon, Player, Move
import re
from reading_mat_games import get_games_from_match_log
import os


def move_list(games_list, pid1, pid2):
    ply = []
    game = []
    games = []
    final_states = []
    for g in games_list:
        # print(g)
        for turn in g:
            if 'final_state' in turn.keys():
                final_states.append(turn['final_state'])
                continue
            for move in turn[1]:
                src_dst = move.split('/')
                # print(src_dst)
                ply.append(Move(pid1, int(src_dst[0]), int(src_dst[1])))
            game.append(ply)
            ply = []
            for move in turn[2]:
                src_dst = move.split('/')
                # print(src_dst)
                ply.append(Move(pid2, int(src_dst[0]), int(src_dst[1])))
            game.append(ply)
            ply = []
        games.append(game)
        game = []
    return games, final_states


def train_on_match(lr, match_file_path, model,
                   train_log, n_games, num_moves_by_result):

    p1 = Player('p1', 'white')
    p2 = Player('p2', 'black')
    game_lists = get_games_from_match_log(match_file_path)
    games, final_states = move_list(game_lists, p1.id, p2.id)
    for i in range(len(games)):
        if len(games[i]) == 0:
            continue
        # for g in games:
        # for i in range(n_matches_sim):
        optimizer = torch.optim.SGD(model.net.parameters(), lr=lr)
        grads = [[torch.zeros((80, 198)), torch.zeros(80), torch.zeros((4, 80)), torch.zeros(4)]
                 for i in range(4)]
        # print(g)
        game = Backgammon(p1, p2)
        game.start_game_deterministic_first_player()
        game_loss = 0
        num_moves = 0
        s = game.encoded_state()
        # display_board(game, game.current_board())
        for ply in games[i]:
            optimizer.zero_grad()
            # if game.is_terminal_state(s):
            # break
            # print([f'move: {p.source()}/{p.destination()}' for p in ply])
            # print(s)
            # register roll based off move
            # print(num_moves)
            game.roll_and_register()
            # move = parse_move(game, ply)

            # display_board(game, game.current_board())
            # out = "Move: "
            # for m in move:
            # out += "({},{}) ".format(m.source(), m.destination())
            # print(out)

            # print(ply)
            copy_game = game.copy()
            copy_game.make_turn_ignore_legality(ply)
            s_1 = copy_game.encoded_state()
            # print(s)
            # print(s_1)
            loss, grads = model.train_single_example(torch.tensor(
                s, requires_grad=True).reshape(-1).type(torch.FloatTensor), torch.tensor(s_1).reshape(-1).type(torch.FloatTensor), optimizer, grads)

            game_loss += loss
            num_moves += 1
            # print(game_loss)
            # print("{},{}\n".format(game.current_player().color, game.move_to_str(move)))
            game.make_turn_ignore_legality(ply)
            s = game.encoded_state()
        final_loss = model.train_final_example(torch.tensor(
            s, requires_grad=True).reshape(-1).type(torch.FloatTensor), torch.tensor(final_states[i]).type(torch.FloatTensor), optimizer, grads)
        game_loss += final_loss
        # print(s)
        train_log.write('{},{},{}\n'.format(n_games,
                                            game_loss/num_moves, final_loss))
        num_moves_by_result[str(final_states[i])] += num_moves
        # state_loader = DataLoader(
        #     [torch.tensor(game.encoded_state()).reshape(-1).type(torch.FloatTensor)], batch_size=4)
        n_games += 1
    return model, n_games, num_moves_by_result


if __name__ == '__main__':

    # adjust these before run
    is_new_output_path = True
    n_games_trained = 0
    new_model = True
    # new_model = False
    # n_games_trained = 118492
    # is_new_output_path = False

    output_path = 'data/tournament_training_results.csv'
    # output_path = 'data/tournament_training_results.csv'
    game_data_path = 'data/tournament_game_data.csv'
    tournament_game_data = open(game_data_path, 'w')

    model_path_format = 'tournament_train_models/td_net_{}_games.pt'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print('Device: {}'.format(device))
    # save_iters = 100
    lambda_param = 0.7
    lr = 0.1
    # lr = 0.025
    model = TD_Net_Wrapper(device=device, lambda_param=lambda_param)
    # model = Gammon_Net_Wrapper(device=device)

    if new_model:
        torch.save(model.net, model_path_format.format(n_games_trained))
    else:
        model.net = torch.load(
            model_path_format.format(n_games_trained)).to(device)

    if is_new_output_path:
        train_log = open(output_path, 'w')
        train_log.write('game,avg_loss,final_loss\n')
    else:
        train_log = open(output_path, 'a')

    num_moves_by_result = {'[1, 0, 0, 0]': 0, '[0, 1, 0, 0]': 0,
                           '[0, 0, 1, 0]': 0, '[0, 0, 0, 1]': 0}

    game_log_dir = 'data/tournament_game_data'

    epochs = 2

    for i in range(epochs):

        for j in range(len(os.listdir(game_log_dir))):
            player_dir = os.listdir(game_log_dir)[j]
            if player_dir[0] == '.':
                continue
            match_log_dir = os.path.join(game_log_dir, player_dir, 'MAT Files')
            for match in os.listdir(match_log_dir):
                if match[-3:] == '.xg' or match[0] == '.':
                    continue
                match_path = os.path.join(match_log_dir, match)
                print(match_path)
            # for match_path in ['data/tournament_game_data/00101 Wolfgang Bacher/MAT Files/001 Kristoffer Hoetzeneder -Wolfgang Bacher_12_2014.mat' for i in range(100)]:
                # for match_path in ['data/tournament_game_data/00101 Wolfgang Bacher/MAT Files/001 Kristoffer Hoetzeneder -Wolfgang Bacher_12_2014.mat']:
                model, n_games_trained, num_moves_by_result = train_on_match(lr, match_path, model, train_log,
                                                                             n_games_trained, num_moves_by_result)

            train_log.close()
            train_log = open(output_path, 'a')
            torch.save(model.net, model_path_format.format(n_games_trained))

    print(num_moves_by_result)
    s1 = ''
    s2 = ''
    for k in num_moves_by_result.keys():
        s1 += f"{k},"
        s2 += f"{num_moves_by_result[k]},"
    tournament_game_data.write(s1 + '\n')
    tournament_game_data.write(s2 + '\n')
    train_log.close()
