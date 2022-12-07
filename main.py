from training import train_network
from self_play import run_selfplay
from utils import SharedStorage, ReplayBuffer
from config import MuZeroConfig, make_chess_config

# MuZero training is split into two independant parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
# TODO add multiprocessing
def muzero(config: MuZeroConfig):
    storage = SharedStorage()
    replay_buffer = ReplayBuffer()

    for _ in range(config.num_actors):
        launch_job(run_selfplay, config, storage, replay_buffer)
    
    train_network(config, storage, replay_buffer)
    return storage.latest_network()

def launch_job(f, *args):
    f(*args)