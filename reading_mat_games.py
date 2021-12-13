import re
import os
import os.path as path
from backgammon import Move

# TODO: Handle "Drops" - player resigns after being offered doubling die
# TODO: Convert move strs to Moves


def parse_moves_from_words(words, new_game):
    move_strs = {1: [], 2: []}
    cp_index = 0
    doubling_factor = 1
    for i in range(len(words)):
        word = words[i]
        if 'Wins' in word:
            points = int(words[i + 1]) / doubling_factor
            if i < 10:
                return {'final_state': [points // 2, int(points == 1), 0, 0]}
            else:
                return {'final_state': [0, 0, points // 2, int(points == 1)]}
        if 'Takes' in word:
            cp_index += 1
            doubling_factor *= 2
        if ':' in word or 'Doubles' in word:
            cp_index += 1
        if '/' in word:
            if word[-1] == ')':
                n_repeats = int(word[-2])
                for i in range(n_repeats):
                    move_strs[cp_index].append(word[:-3])
            else:
                move_strs[cp_index].append(word)
    if move_strs[1] == []:
        move_strs[1].append('-1/-1')
    if move_strs[2] == []:
        move_strs[2].append('-1/-1')
    if new_game and move_strs[2] == ['-1/-1']:
        temp = move_strs[1]
        move_strs[1] = move_strs[2]
        move_strs[2] = temp
    return move_strs


def get_games_from_match_log(path):
    new_game = False
    skip_game = False
    games = []
    game = []
    for l in open(path, 'r').readlines():
        if 'Drops' in l:
            skip_game = True
        l = l.replace('Off', '0').replace('Bar', '25').replace('*', '')
        game_number = 0
        if 'Wins' not in l:
            words = l.strip().split()
        else:
            words = l.split()
        if len(words) == 0 or words[0] == ';':
            continue
        if words[0] == 'Game':
            if not skip_game and len(game) > 0:
                games.append(game)
            game = []
            game_number = int(words[1])
            new_game = True
            skip_game = False
            continue
        if ')' not in words[0]:
            continue
        move_strs = parse_moves_from_words(words, new_game)
        if 'final_state' in move_strs.keys() or move_strs[1] != [] or move_strs[2] != []:
            game.append(move_strs)
        new_game = False
    return games


if __name__ == '__main__':
    game_log_dir = 'data/tournament_game_data'
    player_dirs = os.listdir(game_log_dir)
    game_count = 0

    for player_dir in player_dirs:
        if player_dir[0] == '.':
            continue
        print(player_dir)
        match_log_dir = path.join(game_log_dir, player_dir, 'MAT Files')
        for match_log in os.listdir(match_log_dir):
            if match_log[-3:] == '.xg' or match_log[0] == '.':
                continue
            print(match_log)
            match_log_path = path.join(match_log_dir, match_log)
            games = get_games_from_match_log(match_log_path)
            game_count += len(games)
    print(game_count)
