import numpy as np

class State():
  def: __init__():
    self.white_tiles = new_tiles()
    # 1 for black, -1 for white
    self.current_player = 1
    self.bar = np.zeros(2)


  def new_tiles(self):
    # counts for tiles with black checkers are postive
    # white checkers are negative
    # board is from blacks perspective
    tiles = np.zeros(24)

    tiles[0] = 2
    tiles[5] = -5
    tiles[7] = -3
    tiles[11] = 5
    tiles[12] = -5
    tiles[16] = 3
    tiles[18] = 5
    tiles[23] = -2

    return tiles

  def roll_die(self):
    return np,random.randint(1,7)

  def move_checkers(self, start_position, n):
    pass
  