import copy
import random
import re
import uuid
import numpy as np
import math

standard_board = [(5,5),(8,3),(13,5),(24,2)]

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
    self._bar[color] -=1

  def put_checker_on_bar(self, color, src):
    self._checkers[color][src] -= 1
    self._bar[color] += 1

  def bear_off(self, color, src):
    self._checkers[color][src] -= 1
    self._born_off[color] += 1

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
    self._players = self.ordered_players_with_current_player_id(self._players, current_player_id)
    self._current_player_index = 0
    self._starting_player_id = self.current_player().id
    self._turn_index = 0

  def ordered_players_with_current_player_id(self, players, current_player_id):
    index = next((i for i, player in enumerate(players) if player.id == current_player_id), -1)
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
    roll = lambda : random.randint(1,6)
    pair = (roll(), roll())
    return pair

  def register_roll(self, pair, current_player_id):
    if current_player_id is not None:
      self._rollout[current_player_id].append(pair)

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
    home_board_indices = [self.board_loc_from_point(color, loc) for loc in range(1,6)]
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
    board_dest = self.board_loc_from_point(checker_color, move.destination())
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
    #TODO must make a play if legal one exists
    for move in moves:
      board_src, board_dest = self.board_source_and_dest_from_move(move)
      if move.distance() in unused_rolls:
        unused_rolls.remove(move.distance())
      else:
        return False
      if simul_board.num_checkers_on_bar(checker_color) > 0 and not self.is_entering(move):
        return False
      if self.is_bearing_off(move) and not self.all_checkers_on_home_board(checker):
        return False
      if simul_board.num_checkers_at_index(opp_checker_color, board_dest) > 1:
        return False
      simul_board.move_checker(checker_color, board_src, board_dest)
    return True

  def make_turn(self, moves):
    try:
      assert len(moves) > 0
      roll = self.get_active_roll(self._game_state.current_player().id)
      if self.is_legal_turn(moves, roll):
        checker_color = self._game_state.current_player().color
        opp_checker_color = self._game_state.next_player().color
        for move in moves:
          board_src, board_tgt = self.board_source_and_dest_from_move(move)
          self._board.move_checker(checker_color, board_src, board_tgt)
          if self._board.num_checkers_at_index(opp_checker_color, board_tgt) == 1:
            self._board.put_checker_on_bar(opp_checker_color, board_tgt)
          if self.is_bearing_off(move):
            self._board.bear_off(checker_color, board_src)
          if self.is_entering(move):
            self._board.enter_from_bar(checker_color, board_tgt)
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
      print(current_round)
      print(self._rollout)
      return self._rollout[player_id][current_round]

  def game_is_over(self):
    return False

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
    move_tokens =  re.split(" |;", input_str)
    moves = []
    for token in move_tokens:
      print(token)
      stripped_token = token.strip(")(")

      value_pair = stripped_token.split(',')
      print(value_pair)
      if len(value_pair) != 2:
        raise InvalidMoveFormatException
      move = Move(self._backgammon.current_player().id, int(value_pair[0]), int(value_pair[1]))
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
    print(f"Bar: {curr_player.color}: {n_bar_curr}, {opp_player.color}: {n_bar_opp}")
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
      current_player_checkers = board.num_checkers_at_index(current_player_color, i)
      next_player_checkers = board.num_checkers_at_index(next_player_color, i)
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
    is_valid_player_name = lambda name: len(name)<30 and name.strip() != ""
    while not is_valid_player_name(player_name):
      player_name = str(input("Hello, what is the player's name? (<=30 chars)\n>> "))[:30]

    checker_color = ""
    is_valid_color = lambda color: color in self._checker_colors
    while not is_valid_color(checker_color):
      checker_color = str(input("What color checkers would {} like to play?\nColors: {}\n>> " \
        .format(player_name, self._checker_colors)))
    return player_name, checker_color

  def prompt_for_players(self):
    players = []
    is_unique_player = lambda player_name, checker_color: \
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