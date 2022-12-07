import typing
from typing import Dict, List
from utils import Action

"""A class for the output of the network"""
class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]

# TODO
"""A class to call all the different model"""
class Network(object):
    def initial_inference(self, image) -> NetworkOutput:
        # representation + prediction model
        return NetworkOutput(0, 0, {}, [])
    
    def reccurent_inference(self, hidden_state, action) -> NetworkOutput:
        # dynamics + prediction model
        return NetworkOutput(0, 0, {}, [])
    
    def get_weights(self):
        # Returns the weights of this network
        return []
    
    def training_steps(self):
        # How many steps / batches the network has been trained for.
        return 0

"""Create a new network with uniform weights"""
def make_uniform_network():
    return Network()