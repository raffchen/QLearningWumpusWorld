from typing import Tuple

from .discrete_space import DiscreteSpace


class WumpusWorld:
    def __init__(self):
        self.action_space = DiscreteSpace(0)
        self.observation_space = DiscreteSpace(0)

    @property
    def state(self) -> int:
        return 0

    def step(self, action: int) -> Tuple[int, int, bool]:
        return (0, 0, False)

    def reset(self) -> int:
        return 0
