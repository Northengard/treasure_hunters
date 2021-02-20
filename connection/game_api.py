import zmq
import json


class GameApi:
    def __init__(self, host_addr="tcp://localhost:5556"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(host_addr)
        self.action_space = dict(zip(range(4), ['up', 'right', 'down', 'left']))

    def make_action(self, action):
        action_str = self.action_space[action]
        self.socket.send_string(action_str)
        game_output = self.socket.recv_string()
        game_output = json.loads(game_output)
        return game_output

    def close_connection(self):
        self.socket.close()
        self.context.term()
