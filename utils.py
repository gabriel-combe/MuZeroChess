import collections
from network import Network, make_uniform_network
from game import Player, Game
from config import MuZeroConfig
from typing import List, Optional

MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

"""Dummy class for Chess"""
class MinMaxStats(object):
    """A class that hold the min-max values of the tree"""

    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)
    
    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

# TODO
"""A class for Actions referencement"""
class Action(object):

    def __init__(self, index: int):
        self.index = index
    
    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index
    
    def __gt__(self, other):
        return self.index > other.index

class ActionHistory(object):
    """Simple history container used inside the search.
    
    Onlyused to keep track of the actions executed.
    """

    def __init__(self, history: List[Action], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size
    
    def clone(self):
        return ActionHistory(self.history, self.action_space_size)
    
    def add_action(self, action: Action):
        self.history.append(action)
    
    def last_action(self) -> Action:
        return self.history[-1]
    
    def action_space(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]
    
    def to_play(self) -> Player:
        return Player()

"""A class for a tree"""
class Node(object):

    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0
    
    def expanded(self) -> bool:
        return len(self.children) > 0
    
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

"""A class for the Replay Buffer object"""
class ReplayBuffer(object):
    def __init__(self, config: MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)
    
    # TODO
    def sample_game(self) -> Game:
        #Sample game from buffer either uniformly or according to some priority.
        return self.buffer[0]
    
    # TODO
    def sample_position(self, game) -> int:
        #Sample position from game either uniformly or according to some priority.
        return -1

    
    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [(g.make_image(i), g.history[i:i + num_unroll_steps],
                 g.make_target(i, num_unroll_steps, td_steps, g.to_play())) 
                 for (g, i) in game_pos]

"""A class for the shared storage of all instances"""
class SharedStorage(object):
    def __init__(self):
        self._networks = {}
    
    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return make_uniform_network()
        
    def save_network(self, step: int, network: Network):
        self._networks[step] = network

def softmax_sample(distribution, temprature: float):
    return 0, 0