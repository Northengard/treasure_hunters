import numpy as np
import cv2
from maze_generator import MapGenerator


class GameEmulator(object):
    """
    Main game emulator class

    Attributes:
        field_size: Field size with borders
        ratio: Empty cell to loot ratio
        player_pose: Current in game player's position
        make_step: Dict with actions
        game_map: Matrix containing game map
        agent_state: Information about backpack fillness of player
        render_ratio: Ratio of map render
    """

    def __init__(self, field_size, ratio=10, random_seed=None, render_ratio=4):
        """
        Args:
            field_size: Field size with borders
            ratio: Empty cell to loot ratio
            random_seed: numpy random seed fixation
        """
        if random_seed:
            np.random.seed(random_seed)
        self.ratio = ratio
        self.render_ratio = render_ratio
        self.action_space = dict(zip(range(4), ['left', 'right', 'up', 'down']))
        self.make_step = {0: lambda x: (x[0] - 1, x[1]),  # left
                          1: lambda x: (x[0] + 1, x[1]),  # right
                          2: lambda x: (x[0], x[1] - 1),  # up
                          3: lambda x: (x[0], x[1] + 1)}  # down
        self._treasure_list = list()
        self.map_generator = MapGenerator(map_size=field_size, treasure_prob=0.2, sparsity=2, scale=1)
        self.game_map, self.norm_map = self._generate_map()
        print(len(self._treasure_list))
        self.field_size = self.game_map.shape[0]
        self.player_pose = (self.field_size // 2 + 1, self.field_size // 2 + 2)
        self.agent_state = False
        self._supported_render_types = ['char', 'matrix', 'image']

    def _generate_map(self,  method=1):
        """
        Returns:
            game_map: Matrix field_size x field_size filled by chars, where:
                l - loot,
                f - empty cell,
                e - edge cell,
                s - stock
        """
        if method == 0:
            stock_position = self.make_step[np.random.randint(4)](self.player_pose)

            game_map = np.chararray((self.field_size - 2, self.field_size - 2))
            for i in range(game_map.shape[0]):
                for j in range(game_map.shape[1]):
                    grid_element = np.random.choice(['f'] * self.ratio + ['l'])
                    game_map[i, j] = grid_element
                    if grid_element == 'l':
                        self._treasure_list.append([i, j])
            game_map[self.player_pose] = 'f'
            game_map = np.pad(game_map, pad_width=1, mode='constant', constant_values='e')
            game_map[stock_position] = 's'
            return game_map, game_map.view(dtype=np.uint8)
        elif method == 1:
            game_map, self._treasure_list = self.map_generator.get_maze()
            colors = self.map_generator.get_color_scheme()
            colors = {val: key for key, val in colors.items()}
            char_map = np.chararray(game_map.shape[:2])
            for x in range(game_map.shape[0]):
                for y in range(game_map.shape[1]):
                    char_map[x, y] = colors[tuple(game_map[x, y])]
            return char_map, game_map

    def calc_next_position(self, action):
        """
        Args:
            action: Action number, where:
                0 - up,
                1 - right,
                2 - down,
                3 - left
        Returns:
            Player's position after acting
        """
        return self.make_step[action](self.player_pose)

    def step(self, action):
        """
        Args:
            action: Action number, where:
                0 - up,
                1 - right,
                2 - down,
                3 - left
        Returns:
            Modified matrix(game_map) and reward after making input action
        """
        next_pos = self.calc_next_position(action)
        reward = -1
        if self.agent_state:
            if self.game_map[next_pos] in [b'e']:
                return self.game_map, reward
            elif self.game_map[next_pos] in [b's']:
                self.agent_state = False
                reward = 1
        else:
            if self.game_map[next_pos] == b'e':
                return self.game_map, reward
            elif self.game_map[next_pos] == b'l':
                self.agent_state = True
                self.game_map[next_pos] = 'f'

        self.player_pose = next_pos
        return self.game_map, reward

    def animation_render(self, render_type='char', actions=None):
        pass

    def one_frame_render(self, render_type='char'):  # ToDo: support 'image' render type via openCV based render
        current_map = self.game_map.copy()
        current_map[self.player_pose[::-1]] = 'p'
        if render_type not in self._supported_render_types:
            print(f'render type {render_type} is not supported, use one of:\n{self._supported_render_types}')
            return
        if render_type == 'char':
            print(current_map)
            return current_map
        elif render_type == 'matrix':
            current_map = current_map.view(dtype=np.uint8)
            print(current_map)
            return current_map
        elif render_type == 'image':
            current_map = np.array(current_map.view(dtype=np.uint8))
            current_map[np.where(current_map == 101)] = 0
            current_map[np.where(current_map == 102)] = 255
            current_map[np.where(current_map == 108)] = 128
            current_map[np.where(current_map == 115)] = 55
            current_map[np.where(current_map == 112)] = 200
            current_map = cv2.resize(current_map, None, fx=self.render_ratio, fy=self.render_ratio,
                                     interpolation=cv2.INTER_AREA)

            cv2.imshow('map', current_map)
            cv2.waitKey(300)
        return current_map

    def get_actions(self):
        return self.action_space


if __name__ == '__main__':
    env = GameEmulator(20, random_seed=42, ratio=3, render_ratio=15)
    print(env.get_actions())
    map_ = env.one_frame_render('matrix')
    map_ = env.one_frame_render('image')
    cv2.waitKey(0)
    for act in [0, 3, 0, 1, 2]:
        state, reward = env.step(act)
        map_ = env.one_frame_render('image')
        cv2.waitKey(0)
    # state, reward = env.step(2)
    # map_ = env.one_frame_render('image')
    # state, reward = env.step(2)
    # map_ = env.one_frame_render('image')
    # state, reward = env.step(0)
    # map_ = env.one_frame_render('image')
    # import time
    # time.sleep(10000)
