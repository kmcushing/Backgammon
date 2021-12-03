import copy
import random
import re
import uuid
import numpy as np
import math
import itertools

standard_board = [(5, 5), (8, 3), (13, 5), (24, 2)]


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
        print(color)
        print(src)
        print(dst)
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


class GameState():
    def __init__(self, players=[]):
        print(players)
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
        print(pair)
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

    def is_legal_turn(self, moves, rolls):
        if len([move for move in moves if move.player_id() != self._game_state.current_player().id]) > 0:
            return False

        assert len(rolls) == 2
        unused_rolls = list(rolls).copy()
        if rolls[0] == rolls[1]:
            unused_rolls += unused_rolls
        checker_color = self._game_state.get_player(moves[0].player_id()).color
        opp_checker_color = self._game_state.next_player().color
        simul_board = copy.deepcopy(self._board)
        entered_from_bar = 0
        # TODO must make a play if legal one exists
        for move in moves:
            board_src, board_dest = self.board_source_and_dest_from_move(move)
            # print("src: {}, dest: {}".format(board_src, board_dest))
            if move.distance() in unused_rolls:
                if board_src == self._bar and simul_board.num_checkers_on_bar(checker_color) - entered_from_bar == 0:
                    return False
                elif board_src != self._bar and simul_board.num_checkers_at_index(checker_color, board_src) == 0:
                    return False
                else:
                    unused_rolls.remove(move.distance())
            else:
                if (move.source() == -1 and move.destination() == -1):
                    return True
                elif (sum([simul_board.num_checkers_at_index(checker_color, i) for i in range(6)]) + simul_board.born_off(checker_color) == simul_board.total_checkers()):
                    unused_rolls.sort()
                    for r in unused_rolls:
                        if move.destination() == 0 and (move.source() <= r):
                            unused_rolls.remove(r)
                            continue
                print("a")
                return False
            if simul_board.num_checkers_on_bar(checker_color) - entered_from_bar > 0 and not self.is_entering(move):
                print("b")
                print(move.source())
                print(simul_board.num_checkers_on_bar(checker_color))
                print(not self.is_entering(move))
                return False
            if self.is_bearing_off(move) and not self.all_checkers_on_home_board(self.current_player().color):
                print("c")
                return False
            if simul_board.num_checkers_at_index(opp_checker_color, board_dest) > 1:
                print("d")
                return False
            if self.is_entering(move) and simul_board.num_checkers_on_bar(checker_color) == 0:
                print("e")
                return False
            if self.is_bearing_off(move):
                simul_board.bear_off(checker_color, board_src)
            elif self.is_entering(move):
                print("enter from bar")
                simul_board.enter_from_bar(checker_color, board_dest)
                entered_from_bar += 1
                print(simul_board.num_checkers_at_index(
                    checker_color, board_dest))
            else:
                simul_board.move_checker(checker_color, board_src, board_dest)
            if simul_board.num_checkers_at_index(opp_checker_color, board_dest) == 1:
                simul_board.put_checker_on_bar(opp_checker_color, board_dest)
        return True

    def move_checker(self, move, color):
        board_src, board_dest = self.board_source_and_dest_from_move(move)
        opp_checker_color = self._game_state.next_player().color
        if self.is_bearing_off(move):
            self._board.bear_off(color, board_src)
        elif self.is_entering(move):
            self._board.enter_from_bar(color, board_dest)
        else:
            self._board.move_checker(color, board_src, board_dest)
        if self._board.num_checkers_at_index(opp_checker_color, board_dest) == 1:
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
        return (self._board.born_off(colors[0]) == self._board.total_checkers() / 2
                or self._board.born_off(colors[1]) == self._board.total_checkers() / 2)

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

        valid_actions = []
        curr_player = self._game_state.current_player()
        opp_player = self._game_state.next_player()
        available_checker_points = [point for point in range(1, self._board.board_size(
        )+1) if self._board.num_checkers_at_index(opp_player.color, self.board_loc_from_point(curr_player.color, point)) < 2]
        # only append if can bear off - will need to update after each sim move
        available_checker_points.append(self._off)
        available_srcs = [point for point in range(1, self._board.board_size(
        )+1) if self._board.num_checkers_at_index(curr_player.color, self.board_loc_from_point(curr_player.color, point)) > 0]
        if self._board.num_checkers_on_bar(curr_player.color) > 0:
            available_srcs.append(self._bar)
        rolls = self.get_active_roll(curr_player.id)
        double_roll = rolls[0] == rolls[1]
        move_lengths = ([rolls[0], rolls[0], rolls[0], rolls[0]] if double_roll
                        else [rolls[0], rolls[1]])
        possible_turns = []
        possible_bear_offs = []
        simul_board = copy.deepcopy(self._board)
        num_checkers_moved_to_home_base = 0
        if (sum([self._board.num_checkers_at_index(curr_player.color, i) for i in range(6)]) + self._board.born_off(curr_player.color) + num_checkers_moved_to_home_base == simul_board.total_checkers()):
            for i in range(-5, 0):
                available_checker_points.append(i)
        for src1 in available_srcs:
            for dist1 in move_lengths:
                if (src1 - dist1) in available_checker_points and (src1-dist1) in range(0, self._board.board_size()):
                    move1 = Move(curr_player.id, src1, max(src1-dist1, 0))
                    simul_board2 = copy.deepcopy(simul_board)
                    if (self.board_loc_from_point(curr_player.color, move1.source()) == 25):
                        simul_board2.enter_from_bar(curr_player.color, self.board_loc_from_point(
                            curr_player.color, move1.destination()))
                    else:
                        simul_board2.move_checker(curr_player.color,
                                                  self.board_loc_from_point(curr_player.color, move1.source()), self.board_loc_from_point(curr_player.color, move1.destination()))
                    available_srcs2 = [point for point in range(1, simul_board2.board_size(
                    )+1) if simul_board2.num_checkers_at_index(curr_player.color, self.board_loc_from_point(curr_player.color, point)) > 0]
                    if simul_board2.num_checkers_on_bar(curr_player.color) > 0:
                        available_srcs2.append(self._bar)
                    move_lengths2 = copy.deepcopy(move_lengths)
                    move_lengths2.remove(dist1)
                    if (move1.destination() <= 6 and move1.source() > 6):
                        num_checkers_moved_to_home_base += 1
                    if (sum([self._board.num_checkers_at_index(curr_player.color, i) for i in range(6)]) + self._board.born_off(curr_player.color) + num_checkers_moved_to_home_base == simul_board.total_checkers()):
                        for i in range(-5, 0):
                            available_checker_points.append(i)
                    for src2 in available_srcs2:
                        for dist2 in move_lengths2:
                            if (src2 - dist2) in available_checker_points and (src2-dist2) in range(0, simul_board.board_size()):
                                move2 = Move(curr_player.id, src2,
                                             max(0, src2-dist2))
                                if (not double_roll):
                                    possible_turns.append([move1, move2])
                                else:
                                    simul_board3 = copy.deepcopy(simul_board2)
                                    if (self.board_loc_from_point(curr_player.color, move2.source()) == 25):
                                        simul_board3.enter_from_bar(curr_player.color, self.board_loc_from_point(
                                            curr_player.color, move2.destination()))
                                    else:
                                        simul_board3.move_checker(curr_player.color,
                                                                  self.board_loc_from_point(curr_player.color, move2.source()), self.board_loc_from_point(curr_player.color, move2.destination()))
                                    available_srcs3 = [point for point in range(1, simul_board3.board_size(
                                    )+1) if simul_board3.num_checkers_at_index(curr_player.color, self.board_loc_from_point(curr_player.color, point)) > 0]
                                    move_lengths3 = copy.deepcopy(
                                        move_lengths2)
                                    move_lengths3.remove(dist2)
                                    if (move2.destination() <= 6 and move2.source() > 6):
                                        num_checkers_moved_to_home_base += 1
                                    if (sum([self._board.num_checkers_at_index(curr_player.color, i) for i in range(6)]) + self._board.born_off(curr_player.color) + num_checkers_moved_to_home_base == simul_board.total_checkers()):
                                        for i in range(-5, 0):
                                            available_checker_points.append(i)
                                    for src3 in available_srcs3:
                                        for dist3 in move_lengths3:
                                            if (src3 - dist3) in available_checker_points and (src3-dist3) in range(0, simul_board3.board_size()):
                                                move3 = Move(
                                                    curr_player.id, src3, max(0, src3-dist3))
                                                simul_board4 = copy.deepcopy(
                                                    simul_board3)
                                                if (self.board_loc_from_point(curr_player.color, move3.source()) == 25):
                                                    simul_board4.enter_from_bar(curr_player.color, self.board_loc_from_point(
                                                        curr_player.color, move3.destination()))
                                                else:
                                                    simul_board4.move_checker(curr_player.color,
                                                                              self.board_loc_from_point(curr_player.color, move3.source()), self.board_loc_from_point(curr_player.color, move3.destination()))
                                                available_srcs4 = [point for point in range(1, simul_board4.board_size(
                                                )+1) if simul_board4.num_checkers_at_index(curr_player.color, self.board_loc_from_point(curr_player.color, point)) > 0]
                                                move_lengths4 = copy.deepcopy(
                                                    move_lengths3)
                                                move_lengths4.remove(
                                                    dist3)
                                                if (move3.destination() <= 6 and move3.source() > 6):
                                                    num_checkers_moved_to_home_base += 1
                                                if (sum([self._board.num_checkers_at_index(curr_player.color, i) for i in range(6)]) + self._board.born_off(curr_player.color) + num_checkers_moved_to_home_base == simul_board.total_checkers()):
                                                    for i in range(-5, 0):
                                                        available_checker_points.append(i)
                                                for src4 in available_srcs4:
                                                    for dist4 in move_lengths4:
                                                        if (src4 - dist4) in available_checker_points and (src4-dist4) in range(0, simul_board4.board_size()):
                                                            move4 = Move(
                                                                curr_player.id, src4, max(0, src4-dist4))
                                                            possible_turns.append(
                                                                [move1, move2, move3, move4])

        print("Possibly Legal Moves:")
        for t in possible_turns:
            turn = ""
            for move in t:
                turn += "({},{}) ".format(move.source(), move.destination())
            print(turn)
        possible_turns = [
            turn for turn in possible_turns if self.is_legal_turn(turn, rolls)]
        if len(possible_turns) == 0:
            print("Available checker points: {}".format(
                available_checker_points))
            if -1 in available_checker_points:
                possible_moves = [Move(curr_player.id, src, max(
                    0, src - dist)) for src in range(1, 7) for dist in rolls]
                if not double_roll:
                    possible_turns = [[m] for m in possible_moves if self.is_legal_turn([m], rolls)]
                else:
                    i = 3
                    while len(possible_turns) == 0:
                        possible_turns = itertools.combinations(
                            possible_moves, i)
                        possible_turns = [turn for turn in possible_turns
                                          if self.is_legal_turn(turn, rolls)]
                        i -= 1
            else:
                return [[Move(curr_player.id, -1, -1)]]
        print("Legal Moves:")
        for t in possible_turns:
            turn = ""
            for move in t:
                turn += "({},{}) ".format(move.source(), move.destination())
            print(turn)
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
        if self.current_player_direction() == 1:
            cp_idx = 0
            op_idx = 1
        else:
            cp_idx = 1
            op_idx = 0
        for m in moves:
            encoded_state[cp_idx][m.source()] -= 1
            encoded_state[cp_idx][m.destination()] += 1
            if (encoded_state[op_idx][m.destination()] == 1):
                encoded_state[op_idx][m.destination()] -= 1
                encoded_state[op_idx][25] += 1
        return encoded_state

    def is_terminal_state(self, state):
        return (state[0][0] == self._board.total_checkers() / 2
                or state[1][0] == self._board.total_checkers() / 2)

    def terminal_value(self, state):
        # ASSUMES STATE IS TERMINAL
        if (state[0][0] == self._board.total_checkers / 2):
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


if __name__ == "__main__":
    cli = BackGammonCLI()
    cli.start_prompt()
