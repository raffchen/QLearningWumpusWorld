from typing import Sequence, Tuple

import numpy as np

from .discrete_space import DiscreteSpace


class WumpusWorldSensors:
    """
    Worlds are described by a grid like the following

        00P0
        P00P
        000W
        00PG

    The starting point is always at (0, 0)
    0: empty tile
    P: pit, leads to game over
    W: wumpus, leads to game over
    G: gold, the agent is successful if it reaches here
    """
    class Agent:
        def __init__(self, agentX: int = 0, agentY: int = 0, direction: int = 0,
                     has_arrow: bool = True, has_gold: bool = False):
            self.X = agentX
            self.Y = agentY
            self.direction = direction  # 0: right, 1: down, 2: left, 3: up
            self.has_arrow = has_arrow
            self.has_gold = has_gold

    def __init__(self, custom_board: (Sequence[Sequence[str]], (int, int)) = None):
        """
        the parameter custom_board is a tuple containing the custom world,
        and a tuple containing the x, y coordinates of the wumpus in the custom world
        """
        if custom_board:
            self.board, (self.wumpusX, self.wumpusY) = custom_board
            assert np.array(self.board).shape == (4, 4), "Your custom board must be 4x4"
        else:
            self.board = (
                ('0', '0', 'P', '0'),
                ('P', '0', '0', 'P'),
                ('0', '0', '0', 'W'),
                ('0', '0', 'P', 'G')
            )
            self.wumpusX = 3
            self.wumpusY = 2
        self.agent = self.Agent()

        self.wumpus_killed = False

        self.send_breeze = False
        self.send_bump = False
        self.send_glitter = False
        self.send_scream = False
        self.send_stench = False

        self.num_actions = 6  # forward, turn left, turn right, grab, shoot, climb
        self.num_states = 32  # one for each combination of senses
        self.action_space = DiscreteSpace(self.num_actions)
        self.observation_space = DiscreteSpace(self.num_states)

    @property
    def state(self) -> int:
        for (x, y) in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            new_x = x + self.agent.X
            new_y = y + self.agent.Y

            if 0 <= new_x < 4 and 0 <= new_y < 4:
                if self.board[new_y][new_x] == 'W' and not self.wumpus_killed:
                    self.send_stench = True
                elif self.board[new_y][new_x] == 'P':
                    self.send_breeze = True

        if self.board[self.agent.Y][self.agent.X] == 'G' and not self.gold_taken:
            self.send_glitter = True

        senses = [
            '1' if self.send_stench else '0',
            '1' if self.send_breeze else '0',
            '1' if self.send_glitter else '0',
            '1' if self.send_bump else '0',
            '1' if self.send_scream else '0'
        ]

        self.send_stench = False
        self.send_breeze = False
        self.send_glitter = False
        self.send_bump = False
        self.send_scream = False

        return int(''.join(senses), 2)

    @property
    def gold_taken(self) -> bool:
        return self.agent.has_gold

    def step(self, action: int) -> Tuple[int, int, bool]:
        """
        Returns a tuple in the format of (new state, reward, done)
        given an int, action, where 0 <= action < self.num_actions
        """
        assert 0 <= action < self.num_actions, "Action must be an integer between 0 and 5"

        if action == 0:    # forward
            if self.agent.direction == 0:    # right
                if self.agent.X + 1 > 3:
                    self.send_bump = True
                self.agent.X = min(3, self.agent.X + 1)
            elif self.agent.direction == 1:  # down (towards y = 0)
                if self.agent.Y - 1 < 0:
                    self.send_bump = True
                self.agent.Y = max(0, self.agent.Y - 1)
            elif self.agent.direction == 2:  # left
                if self.agent.X - 1 < 0:
                    self.send_bump = True
                self.agent.X = max(0, self.agent.X - 1)
            else:                            # up
                if self.agent.Y + 1 > 3:
                    self.send_bump = True
                self.agent.Y = min(3, self.agent.Y + 1)

            try:
                if (self.board[self.agent.Y][self.agent.X] == 'P' or
                   (self.board[self.agent.Y][self.agent.X] == 'W' and not self.wumpus_killed)):
                    return (self.state, -1000, True)
                else:
                    return (self.state, -1, False)
            except IndexError as e:
                print(self.agent.X, self.agent.Y)
                raise e
        elif action == 1:  # turn left
            self.agent.direction = (self.agent.direction - 1) % 4
            return (self.state, -1, False)
        elif action == 2:  # turn right
            self.agent.direction = (self.agent.direction + 1) % 4
            return (self.state, -1, False)
        elif action == 3:  # grab
            if self.board[self.agent.Y][self.agent.X] == 'G':
                self.agent.has_gold = True
            return (self.state, -1, False)
        elif action == 4:  # shoot
            if self.agent.has_arrow:
                self.agent.has_arrow = False

                if self.agent.direction == 0:    # right
                    if self.agent.X < self.wumpusX and self.agent.Y == self.wumpusY:
                        self.wumpus_killed = True
                        self.send_scream = True
                elif self.agent.direction == 1:  # down (towards y = 0)
                    if self.agent.X == self.wumpusX and self.agent.Y > self.wumpusY:
                        self.wumpus_killed = True
                        self.send_scream = True
                elif self.agent.direction == 2:  # left
                    if self.agent.X > self.wumpusX and self.agent.Y == self.wumpusY:
                        self.wumpus_killed = True
                        self.send_scream = True
                elif self.agent.direction == 3:  # up
                    if self.agent.X == self.wumpusX and self.agent.Y < self.wumpusY:
                        self.wumpus_killed = True
                        self.send_scream = True
                return (self.state, -10, False)
            else:
                return (self.state, -1, False)
        else:              # climb
            if (self.agent.X, self.agent.Y) == (0, 0) and self.gold_taken:
                return (self.state, 1000, True)
            else:
                return (self.state, -1, False)

    def reset(self, *, agentX: int = 0, agentY: int = 0, direction: int = 0,
              has_arrow: bool = True, has_gold: bool = False, wumpus_killed: bool = False) -> int:
        self.agent = self.Agent(agentX, agentY, direction, has_arrow, has_gold)
        self.wumpus_killed = False
        return 0
