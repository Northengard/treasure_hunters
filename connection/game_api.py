import zmq
import json
import numpy as np
import cv2


class GameAPI:
    def __init__(self, host_addr="tcp://localhost:5556"):
        self._host = host_addr
        self._establish_connection()
        self.action_space = dict(zip(range(4), ['up', 'right', 'down', 'left']))
        self.color_scheme = {101: (20, 20, 20),
                             102: (210, 210, 210),
                             108: (0, 255, 255),
                             115: (0, 166, 255),
                             112: (0, 0, 255)}
        self.treasure_list = list()
        self.render()

    def _establish_connection(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self._host)

    def get_treasure_list(self):
        return self.treasure_list

    def step(self, action):
        action_str = self.action_space.get(action, None)
        if action_str is None:
            raise AssertionError(f'unknown command {action}')
        game_map, player_state, reward, is_done = self._make_request(action_str)
        return game_map, player_state, reward, is_done

    def _create_map(self, game_map, treasures, player):
        game_map[np.where(game_map == 0)] = 101
        treasure_coords = [[treasure['row'], treasure['col']] for treasure in treasures]
        treasure_coords = np.array(treasure_coords)
        game_map[treasure_coords[:, 0], treasure_coords[:, 1]] = 108
        game_map[player['row'], player['col']] = 112
        game_map = game_map[:, ::-1].T

        self.treasure_list = treasure_coords
        self.treasure_list[:, 1] = game_map.shape[0] - 1 - self.treasure_list[:, 1]
        self.treasure_list[:, [0, 1]] = self.treasure_list[:, [1, 0]]
        self.treasure_list = self.treasure_list.tolist()
        return game_map

    def _make_request(self, request_str):
        self.socket.send_string(request_str)
        game_data = self.socket.recv_string()
        game_data = json.loads(game_data)
        game_map = np.array(eval(game_data['staticSceneInfo']))
        treasures = game_data['treasuresInfo']
        player_state = game_data['playerInfo']
        reward = game_data['reward']
        is_done = game_data['isDone']
        game_map = self._create_map(game_map, treasures, player_state)
        return game_map, player_state['isCarry'], reward, is_done

    def render(self, render_type='matrix', visualize=False, wait_key=50):
        action_str = "get_map"
        game_map = self._make_request(action_str)[0]
        return game_map

    def debug_render(self, render_type='matrix', visualize=False, wait_key=50):
        game_map = self.render(render_type, visualize, wait_key)
        if visualize:
            if render_type == 'matrix':
                print(game_map)
            if render_type == 'image':
                image = np.zeros([*game_map.shape, 3])
                for x in range(game_map.shape[0]):
                    for y in range(game_map.shape[1]):
                        image[x, y] = self.color_scheme[game_map[x, y]]
                image = image.astype('uint8')
                current_map = cv2.resize(image, None, fx=20, fy=20,
                                         interpolation=cv2.INTER_AREA)
                cv2.imshow('game_map', current_map)
                cv2.waitKey(wait_key)
        return game_map

    def reset(self):
        self.socket.send_string("restart")
        _ = self.socket.recv_string()
        game_map, player_state, reward, is_done = self._make_request("get_map")
        return game_map, player_state, reward, is_done

    def close_connection(self):
        self.socket.close()
        self.context.term()
