import numpy as np
import cv2
from maze_generator import MapGenerator


class GameEmulator(object):
    """
    Main game emulator class

    Attributes:
        field_size: Field size with borders
        treasure_prob: Empty cell to loot ratio
        player_pose: Current in game player's position
        make_step: Dict with actions
        game_map: Matrix containing game map
        agent_state: Information about backpack fillness of player
        render_ratio: Ratio of map render
    """

    def __init__(self, field_size, treasure_prob=0.2, generation_method=1, sparsity=2,
                 random_seed=None, render_ratio=4):
        """
        Args:
            field_size: Field size with borders
            treasure_prob: Empty cell to loot ratio
            random_seed: numpy random seed fixation
        """
        self.random_seed = random_seed
        if self.random_seed:
            np.random.seed(self.random_seed)
        self.treasure_prob = treasure_prob
        self.render_ratio = render_ratio
        self.action_space = dict(zip(range(4), ['left', 'right', 'up', 'down']))
        self.make_step = {0: lambda x: (x[0], x[1] - 1),  # left
                          1: lambda x: (x[0], x[1] + 1),  # right
                          2: lambda x: (x[0] - 1, x[1]),  # up
                          3: lambda x: (x[0] + 1, x[1])}  # down
        self._treasure_list = list()
        self.map_generator = MapGenerator(map_size=field_size // 2,
                                          treasure_prob=treasure_prob,
                                          sparsity=sparsity,
                                          scale=1,
                                          random_seed=random_seed)
        self.game_map = self._generate_map(generation_method)
        self.field_size = self.game_map.shape[0]
        self.player_pose = (self.field_size // 2 + 2, self.field_size // 2 + 1)
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
            if self.random_seed:
                np.random.seed(self.random_seed)
            stock_position = self.make_step[np.random.randint(4)](self.player_pose)

            game_map = np.chararray((self.field_size - 2, self.field_size - 2))
            for i in range(game_map.shape[0]):
                for j in range(game_map.shape[1]):
                    grid_element = np.random.choice(['f'] * int(100 - self.treasure_prob * 100) +
                                                    ['l'] * int(100 - self.treasure_prob * 100))
                    game_map[i, j] = grid_element
                    if grid_element == 'l':
                        self._treasure_list.append([i, j])
            game_map[self.player_pose] = 'f'
            game_map = np.pad(game_map, pad_width=1, mode='constant', constant_values='e')
            game_map[stock_position] = 's'
            return game_map
        elif method == 1:
            game_map, self._treasure_list = self.map_generator.get_maze()
            colors = self.map_generator.get_color_scheme()
            colors = {val: key for key, val in colors.items()}
            char_map = np.chararray(game_map.shape[:2])
            for x in range(game_map.shape[0]):
                for y in range(game_map.shape[1]):
                    char_map[x, y] = colors[tuple(game_map[x, y])]

            char_map = char_map.astype(str)
            return char_map

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
            if self.game_map[next_pos] in ['e']:
                return self.game_map, reward
            elif self.game_map[next_pos] in ['s']:
                self.agent_state = False
                reward = 1
        else:
            if self.game_map[next_pos] == 'e':
                return self.game_map, reward
            elif self.game_map[next_pos] == 'l':
                self.agent_state = True
                self.game_map[next_pos] = 'f'
                self._treasure_list.remove(list(next_pos))

        self.player_pose = next_pos
        return self.game_map, reward

    def render(self, render_type='char'):
        current_map = self.game_map.copy()
        current_map[self.player_pose] = 'p'
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
            image = np.zeros([*current_map.shape, 3])
            colors = self.map_generator.get_color_scheme()
            for x in range(current_map.shape[0]):
                for y in range(current_map.shape[1]):
                    image[x, y] = colors[current_map[x, y]]
            image = image.astype('uint8')
            current_map = cv2.resize(image, None, fx=self.render_ratio, fy=self.render_ratio,
                                     interpolation=cv2.INTER_AREA)

            cv2.imshow('map', current_map)
            cv2.waitKey(500)
        return current_map

    def get_actions(self):
        return self.action_space


if __name__ == '__main__':
    env = GameEmulator(20, random_seed=42, treasure_prob=0.2, render_ratio=15)
    print(env.get_actions())
    map_ = env.render('char')
    map_ = env.render('image')
    cv2.waitKey(0)
    for act in [0, 3, 0, 1, 2]:
        state, reward = env.step(act)
        map_ = env.render('image')
    # state, reward = env.step(2)
    # map_ = env.one_frame_render('image')
    # state, reward = env.step(2)
    # map_ = env.one_frame_render('image')
    # state, reward = env.step(0)
    # map_ = env.one_frame_render('image')
    # import time
    # time.sleep(10000)
