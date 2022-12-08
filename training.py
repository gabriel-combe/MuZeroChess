import pytorch
from network import Network
from config import MuZeroConfig
from utils import SharedStorage, ReplayBuffer

def train_network(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer):
    network = Network()
    learning_rate = config.lr_init * config.lr_decay_rate**()