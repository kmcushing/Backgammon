import copy
import random
import re
import uuid
import numpy as np
import math
import itertools

standard_board = [(6, 5), (8, 3), (13, 5), (24, 2)]


class Player():
    def __init__(self, name, color):
        self.id = str(uuid.uuid1())
        self.name = name
        self.color = color


class Board():
    def __init__(self, color_1="white", color_2="black"):
        self._board_size = 24
        self._colors = [color_1, color_2]
        self._checkers = {
            color_1: [0]*self._board_size,
            color_2: [0]*self._board_size,
        }
        self._bar = {
            color_1: 0,
            color_2: 0,
        }
        self._born_off = {
            color_1: 0,
            color_2: 0,
        }
        self._total_checkers = 0

    def setup_board(self, board_config, colors):
        for index, checkers in board_config:
            self._checkers[colors[0]][index-1] = checkers
            self._checkers[colors[1]][self.board_size()-index] = checkers
            self._total_checkers += checkers

    def move_checker(self, color, src, dst):
        # print(color)
        # print(src)
        # print(dst)
        self._checkers[color][src] -= 1
        self._checkers[color][dst] += 1

    def enter_from_bar(self, color, dst):
        self._checkers[color][dst] += 1
        self._bar[color] -= 1

    def put_checker_on_bar(self, color, src):
        self._checkers[color][src] -= 1
        self._bar[color] += 1

    def bear_off(self, color, src):
        self._checkers[color][src] -= 1
        self._born_off[color] += 1

    def born_off(self, color):
        return self._born_off[color]

    def num_checkers_at_index(self, color, index):
        if index not in range(self._board_size):
            return 0
        return self._checkers[color][index]

    def num_checkers_on_bar(self, color):
        return self._bar[color]

    def direction_of_color(self, color):
        assert color in self._colors
        return 1 if self._colors.index(color) == 0 else -1

    def board_size(self):
        return self._board_size

    def total_checkers(self):
        return self._total_checkers

    def all_on_home_board(self, color):
        if self.direction_of_color(color) == -1:
            return sum([self.num_checkers_at_index(color, i) for i in range(23,17, -1)]) + self.born_off(color) == self._total_checkers
        return sum([self.num_checkers_at_index(color, i) for i in range(0,6)]) + self.born_off(color) == self._total_checkers

    def copy(self):
        new_board = Board(self._colors[0], self._colors[1])
        new_board._checkers = copy.deepcopy(self._checkers)
        new_board._bar = copy.deepcopy(self._bar)
        new_board._born_off = copy.deepcopy(self._born_off)
        new_board._total_checkers = copy.deepcopy(self._total_checkers)
        return new_board

class Move():
    def __init__(self, player_id, source, destination):
        self._player_id = player_id
        self._source = source
        self._destination = destination
        self._distance = abs(destination - source)

    def source(self):
        return self._source

    def destination(self):
        return self._destination

    def player_id(self):
        return self._player_id

    def distance(self):
        return self._distance

    def __eq__(self, other):
      return self.source() == other.source() and self.destination() == other.destination() and self.distance() == other.distance() and self.player_id() == other.player_id()


class GameState():
    def __init__(self, players=[]):
        # print(players)
        self._players = players
        self._n_players = len(players)
        self._current_player_index = 0
        self._starting_player_id = None
        self._turn_index = 0

    def start_game_with_current_player_id(self, current_player_id):
        self._players = self.ordered_players_with_current_player_id(
            self._players, current_player_id)
        self._current_player_index = 0
        self._starting_player_id = self.current_player().id
        self._turn_index = 0

    def ordered_players_with_current_player_id(self, players, current_player_id):
        index = next((i for i, player in enumerate(players)
                      if player.id == current_player_id), -1)
        try:
            return players[index-1:] + players[:index-1]
        except StopIteration:
            return players

    def next_player_index(self):
        return (self._current_player_index + 1) % self._n_players

    def current_player(self):
        return self._players[self._current_player_index]

    def next_player(self):
        return self._players[self.next_player_index()]

    def valid_move_made(self):
        self._current_player_index = self.next_player_index()
        self._turn_index += 1

    def checker_colors(self):
        return [player.color for player in self._players]

    def get_player(self, player_id):
        return next((player for player in self._players if player.id == player_id), None)

    def current_round(self):
        return math.floor(self._turn_index / self._n_players)

    def starting_player_id(self):
        return self._starting_player_id


class InvalidMoveException(Exception):
    pass


class InvalidMoveFormatException(Exception):
    pass


class NoValidRollException(Exception):
    pass


class Backgammon():
    def __init__(self, player_1, player_2):
        self._board = Board()
        self._game_state = GameState([player_1, player_2])
        self._players = (player_1, player_2)
        self._rollout = {
            player_1.id: [],
            player_2.id: [],
        }
        self._moves = {
            player_1.id: [],
            player_2.id: [],
        }
        self._doubling_die = 1
        self._current_player_index = None
        self._directions = {
            player_1.id: 0,
            player_2.id: 0,
        }
        self._off = 0
        self._bar = 25

    def roll_dice(self):
        def roll(): return random.randint(1, 6)
        pair = (roll(), roll())
        # print(pair)
        return pair

    def register_roll(self, pair, current_player_id):
        if current_player_id is not None:
            self._rollout[current_player_id].append(pair)

    def roll_and_register(self):
        self.register_roll(
            self.roll_dice(), self._game_state.current_player().id)

    def start_game(self):
        pair = self.roll_dice()
        current_player_index = 0 if pair[0] < pair[1] else 1
        current_player = self._players[current_player_index]
        self._game_state.start_game_with_current_player_id(current_player.id)

        colors = self._game_state.checker_colors()
        assert len(colors) >= 2
        self._board = Board(color_1=colors[0], color_2=colors[1])
        self._board.setup_board(standard_board, colors)

    def is_bearing_off(self, move):
        return move.destination() == self._off

    def is_entering(self, move):
        return move.source() == self._bar

    def all_checkers_on_home_board(self, color):
        home_board_indices = [self.board_loc_from_point(
            color, loc) for loc in range(1, 6)]
        return sum([self._board.num_checkers_at_index(color, i) for i in home_board_indices]) \
            == self._board.total_checkers()/2

    def lands_on_valid_point(self, move):
        return self._board.num_checkers_at_index(self.next_player().id, move.destination()) > 1

    def board_loc_from_point(self, color, loc):
        direction = self._board.direction_of_color(color)
        assert abs(direction) == 1
        board_length = self._board.board_size()
        if loc == self._off or loc == self._bar:
            return loc
        else:
            return loc-1 if direction == 1 else board_length-loc

    def board_source_and_dest_from_move(self, move):
        checker_color = self._game_state.get_player(move.player_id()).color
        direction = self._board.direction_of_color(checker_color)
        board_length = self._board.board_size()
        assert abs(direction) == 1
        board_src = self.board_loc_from_point(checker_color, move.source())
        board_dest = self.board_loc_from_point(
            checker_color, move.destination())
        return board_src, board_dest

    def checkers_can_move(self, color, remaining_rolls, board):
        available_dests = [point for point in range(1, board.board_size(
        )+1) if board.num_checkers_at_index(self.next_player().color, self.board_loc_from_point(color, point)) < 2]
        if board.all_on_home_board(color):
          available_dests.append(self._off)
        
        available_srcs = [point for point in range(1, board.board_size(
        )+1) if board.num_checkers_at_index(color, self.board_loc_from_point(color, point)) > 0]
        if board.num_checkers_on_bar(color) > 0:
            available_srcs.append(self._bar)

        # print("Available srcs: {}".format(available_srcs))
        # print("Available dests: {}".format(available_dests))
        if sum([(max(0, src - dist)) in available_dests for dist in remaining_rolls for src in available_srcs]) > 0:
            return True
        return False
        

    def is_legal_turn(self, moves, rolls):
        # out = 'Move: '
        # for m in moves:
        #     out += '({},{}) '.format(m.source(), m.destination())
        # print(out)
        if len([move for move in moves if move.player_id() != self._game_state.current_player().id]) > 0:
            # print("wrong pid")
            return False

        assert len(rolls) == 2
        unused_rolls = list(rolls).copy()
        if rolls[0] == rolls[1]:
            unused_rolls += unused_rolls
        checker_color = self._game_state.current_player().color
        opp_checker_color = self._game_state.next_player().color
        simul_board = self._board.copy()
        for move in moves:
            board_src, board_dest = self.board_source_and_dest_from_move(move)
            # print("src: {}, dest: {}".format(board_src, board_dest))
            if move.distance() in unused_rolls:
                if board_src == self._bar and simul_board.num_checkers_on_bar(checker_color) == 0:
                    # print("no checkers on bar")
                    return False
                elif board_src != self._bar and simul_board.num_checkers_at_index(checker_color, board_src) == 0:
                    # print("no checkers at index")
                    return False
                else:
                    unused_rolls.remove(move.distance())
            else:
                if (move.source() == -1 and move.destination() == -1):
                    return True
                elif simul_board.all_on_home_board(checker_color):
                    unused_rolls.sort()
                    move_successful = False
                    # print(simul_board.num_checkers_at_index(checker_color, board_src))
                    if simul_board.num_checkers_at_index(checker_color, board_src) <= 0:
                        # print("no checkers at index when bearing off")
                        return False
                    for r in unused_rolls:
                        if move.destination() == 0 and (move.source() <= r):
                            unused_rolls.remove(r)
                            move_successful = True
                            break
                    if move_successful:
                        simul_board.bear_off(checker_color, board_src)
                        continue
                # print(unused_rolls)
                # print("a")
                # print("illegal move dist")
                return False
            if simul_board.num_checkers_at_index(checker_color, board_src) < 1 and board_src != self._bar:
                # print("no checkers to move")
                return False
            if simul_board.num_checkers_on_bar(checker_color) < 1 and board_src == self._bar:
                # print("no checkers on bar to enter")
                return False
            # print(simul_board.num_checkers_on_bar(checker_color))
            # print(self.is_entering(move))
            if simul_board.num_checkers_on_bar(checker_color) > 0 and not self.is_entering(move):
                # print("b")
                # print(move.source())
                # print(simul_board.num_checkers_on_bar(checker_color))
                # print(not self.is_entering(move))
                # print("on bar but not entering")
                return False
            if self.is_bearing_off(move) and not simul_board.all_on_home_board(checker_color):
                # print("c")
                # print("not on home board - can't bear off")
                return False
            if simul_board.num_checkers_at_index(opp_checker_color, board_dest) > 1 and not self.is_bearing_off(move):
                # print("d")
                # print("opponent controls dest")
                return False
            if self.is_entering(move) and simul_board.num_checkers_on_bar(checker_color) == 0:
                # print("e")
                # print("no checkers to enter from bar")
                return False
            if self.is_bearing_off(move):
                simul_board.bear_off(checker_color, board_src)
            elif self.is_entering(move):
                # print("enter from bar")
                simul_board.enter_from_bar(checker_color, board_dest)
                # print(simul_board.num_checkers_at_index(
                #     checker_color, board_dest))
            else:
                simul_board.move_checker(checker_color, board_src, board_dest)
            if simul_board.num_checkers_at_index(opp_checker_color, board_dest) == 1:
                simul_board.put_checker_on_bar(opp_checker_color, board_dest)
        if len(unused_rolls) > 0 and self.checkers_can_move(checker_color, unused_rolls, simul_board):
            # print("not enough moves")
            return False
        return True

    def move_checker(self, move, color):
        board_src, board_dest = self.board_source_and_dest_from_move(move)
        opp_checker_color = self._game_state.next_player().color
        if self.is_bearing_off(move):
            self._board.bear_off(color, board_src)
        else:
            if self.is_entering(move):
                self._board.enter_from_bar(color, board_dest)
            else:
                self._board.move_checker(color, board_src, board_dest)
            if self._board.num_checkers_at_index(opp_checker_color, board_dest) == 1 and not self.is_bearing_off(move):
                self._board.put_checker_on_bar(opp_checker_color, board_dest)

    def make_turn(self, moves):
        try:
            assert len(moves) > 0
            roll = self.get_active_roll(self._game_state.current_player().id)
            if self.is_legal_turn(moves, roll):
                checker_color = self._game_state.current_player().color
                for move in moves:
                    if (move.source() != -1 and move.destination() != -1):
                        self.move_checker(move, checker_color)
                self._game_state.valid_move_made()
            else:
                raise InvalidMoveException()
        except NoValidRollException as e:
            print(e)

    def get_active_roll(self, player_id):
        current_round = self._game_state.current_round()
        if len(self._rollout[player_id]) < current_round:
            raise NoValidRollException()
        else:
            # print(current_round)
            # print(self._rollout)
            return self._rollout[player_id][current_round]

    def game_is_over(self):
        colors = self._game_state.checker_colors()
        return (self._board.born_off(colors[0]) >= self._board.total_checkers()
                or self._board.born_off(colors[1]) >= self._board.total_checkers())

    def valid_actions(self):
        # returns a list of lists
        # each inner list is a list of moves constituting a valid turn

        # valid_actions = []
        # curr_player = self._game_state.current_player()
        # opp_player = self._game_state.next_player()
        # available_checker_points = [point for point in range(1, self._board.board_size(
        # )+1) if self._board.num_checkers_at_index(opp_player.color, point) < 2]
        # available_checker_points.append(self._bar)
        # available_checker_points.append(self._off)
        # rolls = self.get_active_roll(curr_player.id)
        # double_roll = rolls[0] == rolls[1]
        # possible_moves = []
        # print(rolls)
        # print(available_checker_points)
        # for src in available_checker_points:
        #     for distance in rolls:
        #         if (src - distance) in available_checker_points and (src-distance) in range(0, self._board.board_size()):
        #             move = Move(curr_player.id, src, src-distance)
        #             possible_moves.append(move)
        #             if double_roll:
        #                 for _ in range(3):
        #                     possible_moves.append(move)
        # print("going into possible turns")
        # print(len(list(itertools.combinations(
        #     possible_moves, 4 if double_roll else 2))))
        # possible_turns = [turn for turn in itertools.combinations(
        #     possible_moves, 4 if double_roll else 2) if self.is_legal_turn(turn, rolls)]

        curr_player = self._game_state.current_player()
        opp_player = self._game_state.next_player()
        available_checker_points = [point for point in range(1, self._board.board_size(
        )+1) if self._board.num_checkers_at_index(opp_player.color, self.board_loc_from_point(curr_player.color, point)) < 2]
        # only append if can bear off - will need to update after each sim move
        # print("all home: {}".format(self._board.all_on_home_board(curr_player.color)))
        if self._board.all_on_home_board(curr_player.color):
          available_checker_points.append(self._off)
        available_srcs = [point for point in range(1, self._board.board_size(
        )+1) if self._board.num_checkers_at_index(curr_player.color, self.board_loc_from_point(curr_player.color, point)) > 0]
        if self._board.num_checkers_on_bar(curr_player.color) > 0:
            available_srcs.append(self._bar)
        rolls = self.get_active_roll(curr_player.id)
        double_roll = rolls[0] == rolls[1]
        move_lengths = ([rolls[0], rolls[0], rolls[0], rolls[0]] if double_roll
                        else [rolls[0], rolls[1]])
        # possible_turns = []
        # possible_bear_offs = []
        simul_board = copy.deepcopy(self._board)
        
        def possibly_legal_sequences(curr_player, opp_player, board, move_lengths, move_seqs, current_move_seq):
            if len(move_lengths) == 0:
                # maybe add or all checkers beared off here?
                # move_seqs.append(current_move_seq)
                return move_seqs

            simul_board = board.copy()

            available_dests = [point for point in range(1, simul_board.board_size(
            )+1) if simul_board.num_checkers_at_index(opp_player.color, self.board_loc_from_point(curr_player.color, point)) < 2]
            # only append if can bear off - will need to update after each sim move
            if simul_board.all_on_home_board(curr_player.color):
                for i in range(-5,1):
                  available_dests.append(i)
                available_checker_points.append(self._off)
            # print(available_dests)
            available_srcs = [point for point in range(1, simul_board.board_size(
            )+1) if simul_board.num_checkers_at_index(curr_player.color, self.board_loc_from_point(curr_player.color, point)) > 0]
            if simul_board.num_checkers_on_bar(curr_player.color) > 0:
                available_srcs.append(self._bar)
            # print(f"srcs: {available_srcs}")
            # print(f"dests: {available_dests}")
            for src in available_srcs:
                for dist in move_lengths:
                    if (src - dist) in available_dests and (src-dist) in range(min(0, *available_dests), simul_board.board_size() + 1):
                        move = Move(curr_player.id, src, max(src-dist, 0))
                        move_lengths2 = copy.deepcopy(move_lengths)
                        move_lengths2.remove(dist)
                        depth = len(current_move_seq)
                        if depth > 1:
                          if sum([[move, current_move_seq[-1]] == seq[depth - 2:depth] and len(seq) > depth for seq in move_seqs]) > 0 and current_move_seq[-1].source() != 25 and move.destination() > 0 and simul_board.born_off(curr_player.color) < 1:
                            # print("Move: ({},{})".format(move.source(), move.destination()))
                            # print('skip this combo')
                            continue
                        current_move_seq2 = copy.deepcopy(current_move_seq)
                        current_move_seq2.append(move)
                        simul_board2 = simul_board.copy()
                        move_seqs.append(current_move_seq2)
                        if (self.is_entering(move)):
                            # move_seqs.append(current_move_seq2)
                            simul_board2.enter_from_bar(curr_player.color, self.board_loc_from_point(
                                curr_player.color, move.destination()))
                        elif (self.is_bearing_off(move)):
                            # print("appending moves of length {}".format(len(current_move_seq2)))
                            # move_seqs.append(current_move_seq2)
                            simul_board2.bear_off(curr_player.color, self.board_loc_from_point(curr_player.color, src))
                        else:
                            simul_board2.move_checker(curr_player.color,
                                                    self.board_loc_from_point(curr_player.color, src), self.board_loc_from_point(curr_player.color, move.destination()))
                            # board_src, board_dst = self.board_source_and_dest_from_move(move)
                        move_seqs = possibly_legal_sequences(curr_player, 
                            opp_player, simul_board2, move_lengths2, move_seqs, current_move_seq2)
            return move_seqs


        possible_turns = possibly_legal_sequences(curr_player, opp_player, self._board, move_lengths, [],[])

            

        # print("Possibly Legal Moves:")
        # for t in possible_turns:
        #     turn = ""
        #     for move in t:
        #         turn += "({},{}) ".format(move.source(), move.destination())
        #     print(turn)
        possible_turns = [
            turn for turn in possible_turns if self.is_legal_turn(turn, rolls)]
        if len(possible_turns) == 0:
            # print("Available checker points: {}".format(
                # available_checker_points))
            # print(available_srcs)
            if 0 in available_checker_points:
                # available_srcs += [i for in range(1, 7)]
                # print(rolls)
                possible_moves = [Move(curr_player.id, src, max(
                    0, src - dist)) for src in  available_srcs for dist in rolls]
                if not double_roll:
                    possible_turns = [[m1, m2] for m1, m2 in itertools.combinations(possible_moves, 2) if self.is_legal_turn([m1, m2], rolls)]
                else:
                    possible_moves += possible_moves
                    possible_turns = [[m1, m2, m3, m4] for m1, m2, m3, m4 in itertools.permutations(possible_moves, 4) if self.is_legal_turn([m1, m2, m3, m4], rolls)]
                    # i = 3
                    # while len(possible_turns) == 0:
                    #     possible_turns = itertools.combinations(
                    #         possible_moves, i)
                    #     possible_turns = [turn for turn in possible_turns
                    #                       if self.is_legal_turn(turn, rolls)]
                    #     i -= 1
            if len(possible_turns) < 1:
                possible_turns =[[Move(curr_player.id, -1, -1)]]
        # print("Legal Moves:")
        # for t in possible_turns:
        #     turn = ""
        #     for move in t:
        #         turn += "({},{}) ".format(move.source(), move.destination())
        #     print(turn)
        return possible_turns

    def encoded_state(self):
        state = np.zeros((2, 26))
        cp_board = copy.deepcopy(
            self._board._checkers[self.current_player().color])
        op_board = copy.deepcopy(
            self._board._checkers[self.next_player().color])
        if self._board.direction_of_color(self.current_player().color) == 1:
            op_board.reverse()
            cp_idx = 0
            op_idx = 1
        else:
            cp_board.reverse()
            cp_idx = 1
            op_idx = 0

        state[cp_idx][0] = self._board._born_off[self.current_player().color]
        state[cp_idx][1:25] = cp_board
        state[cp_idx][25] = self._board._bar[self.current_player().color]
        state[op_idx][0] = self._board._born_off[self.next_player().color]
        state[op_idx][1:25] = op_board
        state[op_idx][25] = self._board._bar[self.next_player().color]

        return state

    def current_player_direction(self):
        return self._board.direction_of_color(self.current_player().color)

    def simulate_single_move(self, encoded_state, moves):
        # ASSUMES MOVE IS VALID
        if (moves[0].source() == -1 and moves[0].destination() == -1):
            return encoded_state
        encoded_state = copy.deepcopy(encoded_state)
        if self.current_player_direction() == 1:
            cp_idx = 0
            op_idx = 1
        else:
            cp_idx = 1
            op_idx = 0
        for m in moves:
            if m.source() != 25:
                encoded_state[cp_idx][self.board_loc_from_point(self._game_state.checker_colors()[0], m.source()) + 1] -= 1
            else: 
                encoded_state[cp_idx][25] -= 1
            if m.destination() != 0:
                encoded_state[cp_idx][self.board_loc_from_point(self._game_state.checker_colors()[0], m.destination()) + 1] += 1
                if (encoded_state[op_idx][self.board_loc_from_point(self._game_state.checker_colors()[0], m.destination()) + 1] == 1):
                    encoded_state[op_idx][self.board_loc_from_point(self._game_state.checker_colors()[0], m.destination()) + 1] -= 1
                    encoded_state[op_idx][25] += 1
            else:
                encoded_state[cp_idx][0] += 1
        return encoded_state

    def is_terminal_state(self, state):
        return (state[0][0] == self._board.total_checkers()
                or state[1][0] == self._board.total_checkers())

    def terminal_value(self, state):
        # ASSUMES STATE IS TERMINAL
        if (state[0][0] == self._board.total_checkers()):
            if state[1][0] == 0:
                return [1, 0, 0, 0]
            else:
                return [0, 1, 0, 0]
        if state[0][0] == 0:
            return [0, 0, 0, 1]
        return [0, 0, 1, 0]

    def current_board(self):
        return self._board

    def current_player_name(self):
        return self._game_state.current_player().name

    def current_player(self):
        return self._game_state.current_player()

    def next_player(self):
        return self._game_state.next_player()
    
    def move_to_str(self, moves):
        move_str = ''
        if moves[0].source() == -1 and moves[0].destination() == -1:
            return ''
        for i in range(len(moves)):
            if i > 0:
                move_str += '/'
            move_str += '{}-{}'.format(moves[i].source(), moves[i].destination())
        return move_str


class BackGammonCLI():
    def __init__(self):
        self._checker_colors = ["white", "black"]
        self._color_to_letter = {
            self._checker_colors[0]: 'X',
            self._checker_colors[1]: 'O',
        }

    def is_unique_player(self, players, new_player):
        names = [player.name for player in players]
        colors = [player.color for player in players]
        return new_player.name not in names and new_player.color not in colors

    def generate_moves(self, input_str, current_player):
        move_tokens = re.split(" |;", input_str)
        moves = []
        for token in move_tokens:
            print(token)
            stripped_token = token.strip(")(")

            value_pair = stripped_token.split(',')
            print(value_pair)
            if len(value_pair) != 2:
                raise InvalidMoveFormatException
            move = Move(self._backgammon.current_player().id,
                        int(value_pair[0]), int(value_pair[1]))
            moves.append(move)
        return moves

    def prompt_turn(self):
        moves = []
        valid_move_made = False
        curr_player = self._backgammon.current_player()
        opp_player = self._backgammon._game_state.next_player()
        print(f"It is {curr_player.name}'s turn")
        rolls = self._backgammon.roll_dice()
        self._backgammon.register_roll(rolls, curr_player.id)
        print(f"Rolls: {rolls}")
        n_bar_curr = self._backgammon.current_board().num_checkers_on_bar(curr_player.color)
        n_bar_opp = self._backgammon.current_board().num_checkers_on_bar(opp_player.color)
        print(
            f"Bar: {curr_player.color}: {n_bar_curr}, {opp_player.color}: {n_bar_opp}")
        while not valid_move_made:
            prompt = "Enter your moves in the form (source, destination) eg. (2,4) (4,8)\n>> "
            response = str(input(prompt))
            try:
                moves = self.generate_moves(response, curr_player)
                self._backgammon.make_turn(moves)
                valid_move_made = True
            except InvalidMoveException as e:
                print("Invalid move made, try again")
            except InvalidMoveFormatException:
                print("Invalid Move Format")

    def display_board(self, board):
        point_row = ""
        point_label_row = ""
        current_player = self._backgammon.current_player()
        current_player_color = current_player.color
        next_player_color = self._backgammon.next_player().color
        current_player_started = self._backgammon._game_state.starting_player_id() == current_player.id
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
        print(f"current player direction: <<")

    def print_player_info(self, players):
        for player in players:
            print(f'Name: {player.name}\n')
            print(f'Color: {player.color}\n')
            print(f'Letter: {self._color_to_letter[player.color]}')

    def prompt_for_player_info(self):
        player_name = ""
        def is_valid_player_name(name): return len(
            name) < 30 and name.strip() != ""
        while not is_valid_player_name(player_name):
            player_name = str(
                input("Hello, what is the player's name? (<=30 chars)\n>> "))[:30]

        checker_color = ""
        def is_valid_color(color): return color in self._checker_colors
        while not is_valid_color(checker_color):
            checker_color = str(input("What color checkers would {} like to play?\nColors: {}\n>> "
                                      .format(player_name, self._checker_colors)))
        return player_name, checker_color

    def prompt_for_players(self):
        players = []
        def is_unique_player(player_name, checker_color): return \
            player_name not in [player.color for player in players]
        while len(players) < 2:
            player_name, checker_color = self.prompt_for_player_info()
            new_player = Player(player_name, checker_color)
            if is_unique_player(players, new_player):
                players.append(new_player)
        return players

    def start_prompt(self):
        players = self.prompt_for_players()
        self._backgammon = Backgammon(players[0], players[1])
        self._backgammon.start_game()
        self.print_player_info(players)

        while not self._backgammon.game_is_over():
            self.display_board(self._backgammon.current_board())
            self.prompt_turn()

        print("Game Over")

class Sim():
    def __init__(self, path):
        self.f = open(path, 'r')
        first_moves = self.f.readlines()[0:2]
        self.f.seek(0)
        colors = [move.strip().split(",")[0] for move in first_moves]
        print(colors)
        self.game = Backgammon(Player('p1', colors[0]), Player('p2', colors[1]))
        self.game.start_game()
    
    def sim_to_move(self, num):
        lines = self.f.readlines()
        for i in range(1, num + 1):
            l = lines[i]
            vals = l.split(',')[1].split('/')
            move_vals = []
            for move in vals:
                move_vals.append(move.strip().split('-'))
            print(f"move number {i}")
            moves = []
            if move_vals == [['']]:
                self.game.register_roll((1,1), self.game.current_player().id)
                moves.append(Move(self.game.current_player().id, -1, -1))
            else:
                for m in move_vals:
                    moves.append(Move(self.game.current_player().id, int(m[0]), int(m[1])))
                self.game.register_roll((moves[0].source() - moves[0].destination(), moves[1].source() - moves[1].destination()), self.game.current_player().id)
            self.game.make_turn(moves)
            display_board(self.game, self.game._board)
            move_str = "Move: "
            for m in moves:
                move_str += f"({m.source()},{m.destination()}) "
            print(move_str)

# if __name__ == "__main__":
#     cli = BackGammonCLI()
#     cli.start_prompt()

def display_board(self, board):
        point_row = ""
        point_label_row = ""
        current_player = self.current_player()
        current_player_color = self.current_player().color
        next_player_color = self.next_player().color
        current_player_started = self._game_state.starting_player_id() == current_player.id
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
        n_bar_curr = self.current_board().num_checkers_on_bar(self.current_player().color)
        n_bar_opp = self.current_board().num_checkers_on_bar(self.next_player().color)
        print(
            f"Bar: {self.current_player().color}: {n_bar_curr}, {self.next_player().color}: {n_bar_opp}")
        print(color_key)
        print(f"current player direction: <<")

if __name__ == "__main__":
    sim = Sim('data/game_logs/game_0_log.csv')
    sim.sim_to_move(38)
