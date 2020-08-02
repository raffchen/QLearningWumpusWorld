from typing import Tuple

from .discrete_space import DiscreteSpace


class WumpusWorld:
    """
    Worlds are described by a grid like the following

        00P0
        P00P
        0000
        W0PG

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

    def __init__(self):
        self.board = [
            ['0', '0', 'P', '0'],
            ['P', '0', '0', 'P'],
            ['0', '0', '0', '0'],
            ['W', '0', 'P', 'G']
        ]
        self.agent = self.Agent()
        self.wumpusX = 0
        self.wumpusY = 3

        self.num_actions = 6  # forward, turn left, turn right, grab, shoot, climb
        self.num_states = 16  # one for each tile
        self.action_space = DiscreteSpace(self.num_actions)
        self.observation_space = DiscreteSpace(self.num_states)

    @property
    def state(self) -> int:
        return self.agent.Y * 4 + self.agent.X + \
            (16 * int(f"{bin(self.agent.direction)}{int(self.agent.has_arrow)}{int(self.agent.has_gold)}", 2))

    def step(self, action: int) -> Tuple[int, int, bool]:
        """
        Returns a tuple in the format of (new state, reward, done)
        given an int, action, where 0 <= action < self.num_actions
        """
        assert 0 <= action < self.num_actions, "Action must be an integer between 0 and 5"

        if action == 0:    # forward
            if self.agent.direction == 0:    # right
                self.agent.X = min(3, self.agent.X + 1)
            elif self.agent.direction == 1:  # down (towards y = 0)
                self.agent.Y = max(0, self.agent.Y - 1)
            elif self.agent.direction == 2:  # left
                self.agent.X = max(0, self.agent.X + 1)
            else:                            # up
                self.agent.Y = min(3, self.agent.Y + 1)

            if self.board[self.agent.Y][self.agent.X] in ('P', 'W'):
                return (self.state, -1000, True)
            else:
                return (self.state, -1, False)
        elif action == 1:  # turn left
            self.agent.direction = (self.agent.direction - 1) % 4
            return (self.state, -1, False)
        elif action == 2:  # turn right
            self.agent.direction = (self.agent.direction + 1) % 4
            return (self.state, -1, False)
        elif action == 3:  # grab
            if self.board[self.agent.Y][self.agent.X] == 'G':
                self.agent.has_gold = True
                self.board[self.agent.Y][self.agent.X] = '0'
            return (self.state, -1, False)
        elif action == 4:  # shoot
            if self.agent.has_arrow:
                self.agent.has_arrow = False

                if self.agent.direction == 0:    # right
                    if self.agent.X < self.wumpusX and self.agent.Y == self.wumpusY:
                        self.board[self.wumpusY][self.wumpusX] = '0'
                elif self.agent.direction == 1:  # down (towards y = 0)
                    if self.agent.X == self.wumpusX and self.agent.Y > self.wumpusY:
                        self.board[self.wumpusY][self.wumpusX] = '0'
                elif self.agent.direction == 2:  # left
                    if self.agent.X > self.wumpusX and self.agent.Y == self.wumpusY:
                        self.board[self.wumpusY][self.wumpusX] = '0'
                elif self.agent.direction == 3:  # up
                    if self.agent.X == self.wumpusX and self.agent.Y < self.wumpusY:
                        self.board[self.wumpusY][self.wumpusX] = '0'
                return (self.state, -10, False)
            else:
                return (self.state, -1, False)
        else:              # climb
            if (self.agent.X, self.agent.Y) == (0, 0) and self.agent.has_gold:
                return (self.state, 1000, True)
            else:
                return (self.state, -1, False)

    def reset(self) -> int:
        # TODO: allow resetting to different points in the world
        return 0
