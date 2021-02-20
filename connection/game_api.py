import zmq
import json
import numpy as np


class GameAPI:
    def __init__(self, host_addr="tcp://localhost:5556"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(host_addr)
        self.action_space = dict(zip(range(4), ['up', 'right', 'down', 'left']))

    def step(self, action):
        action_str = self.action_space[action]
        self.socket.send_string(action_str)
        game_map = self.socket.recv_string()
        game_map = np.array(json.loads(game_map))
        return game_map

    def render(self, render_type):
        pass

    def reset(self):
        pass

    def close_connection(self):
        self.socket.close()
        self.context.term()
