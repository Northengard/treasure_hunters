import sys
from time import perf_counter
import numpy as np
import cv2
from .maze_generator import MapGenerator


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

    def __init__(self, field_size, treasure_prob=0.2, generation_method='empty', sparsity=2,
                 random_seed=None, render_ratio=4, max_game_steps=1000):
        """
        Args:
            field_size: Field size with borders
            treasure_prob: Empty cell to loot ratio
            generation_method: 'empty' for 'no walls' method and or 'maze' for maze generation
            sparsity: maze generator parameter, affects
            random_seed: numpy random seed fixation
            render_ratio: render scale coef
            max_game_steps: maximum step in the game
        """
        self.random_seed = random_seed
        if self.random_seed:
            np.random.seed(self.random_seed)
        self.treasure_prob = treasure_prob
        self.render_ratio = render_ratio
        self.action_space = dict(zip(range(4), ['up', 'right', 'down', 'left']))
        self.make_step = {0: lambda x: (x[0] - 1, x[1]),  # up
                          1: lambda x: (x[0], x[1] + 1),  # right
                          2: lambda x: (x[0] + 1, x[1]),  # down
                          3: lambda x: (x[0], x[1] - 1)}  # left
        self._generation_methods = {'empty': 0, 'maze': 1}
        generation_method = self._generation_methods.get(generation_method, None)
        if not generation_method:
            raise AssertionError(f"invalid generation method: use one of: {self._generation_methods.keys()}")
        self._treasure_list = list()
        self.field_size = field_size + 2
        self.player_pose = (self.field_size // 2 + 1, self.field_size // 2)
        self.map_generator = MapGenerator(map_size=field_size // 2,
                                          treasure_prob=treasure_prob,
                                          sparsity=sparsity,
                                          scale=1,
                                          random_seed=random_seed)

        self.generation_method = generation_method
        self.game_map = self._generate_map(generation_method)

        self.agent_state = False
        self._supported_render_types = ['char', 'matrix', 'image']
        self.max_game_steps = max_game_steps
        self.current_game_step = 0

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

            game_map = np.zeros((self.field_size - 2, self.field_size - 2)).astype(str)
            for i in range(game_map.shape[0]):
                for j in range(game_map.shape[1]):
                    grid_element = np.random.choice(['f'] * int(100 - self.treasure_prob * 100) +
                                                    ['l'] * int(self.treasure_prob * 100))
                    game_map[i, j] = grid_element
                    if grid_element == 'l':
                        self._treasure_list.append([i+1, j+1])
            game_map[self.player_pose] = 'f'
            game_map = np.pad(game_map, pad_width=1, mode='constant', constant_values='e')
            game_map[stock_position] = 's'
            return game_map
        elif method == 1:
            game_map, self._treasure_list = self.map_generator.get_maze()
            colors = self.map_generator.get_color_scheme()
            colors = {val: key for key, val in colors.items()}
            char_map = np.zeros(game_map.shape[:2]).astype(str)
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

        if self.max_game_steps <= self.current_game_step:
            is_done = True
        else:
            self.current_game_step = self.current_game_step + 1
            is_done = False

        next_pos = self.calc_next_position(action)
        reward = -1

        if self.game_map[next_pos] != 'e':
            if self.game_map[next_pos] == 's':
                if self.agent_state:
                    self.agent_state = False
                    reward = 1
            elif self.game_map[next_pos] == 'l':
                if not self.agent_state:
                    self.agent_state = True
                    self.game_map[next_pos] = 'f'
                    self._treasure_list.remove(list(next_pos))

            self.player_pose = next_pos

        return self.render(render_type='matrix'), self.agent_state, reward, is_done

    def reset(self):
        """
        reset enviroment to the start state
        Returns:

        """
        self._treasure_list = list()
        self.player_pose = (self.field_size // 2 + 2, self.field_size // 2 + 1)
        self.game_map = self._generate_map(self.generation_method)
        self.agent_state = False
        is_done = False
        return self.render(render_type='matrix'), self.agent_state, is_done

    def render(self, render_type='char', visualize=False, wait_key=50):
        current_map = self.game_map.copy()
        current_map[self.player_pose] = 'p'
        if render_type not in self._supported_render_types:
            print(f'render type {render_type} is not supported, use one of:\n{self._supported_render_types}')
            return
        elif render_type == 'matrix':
            current_map = np.vectorize(ord)(current_map)
        elif render_type == 'image':
            image = np.zeros([*current_map.shape, 3])
            colors = self.map_generator.get_color_scheme()
            for x in range(current_map.shape[0]):
                for y in range(current_map.shape[1]):
                    image[x, y] = colors[current_map[x, y]]
            image = image.astype('uint8')
            current_map = cv2.resize(image, None, fx=self.render_ratio, fy=self.render_ratio,
                                     interpolation=cv2.INTER_AREA)

            if visualize:
                cv2.imshow('map', current_map)
                cv2.waitKey(wait_key)
        if visualize and render_type != 'image':
            print(current_map)
        return current_map

    def get_actions(self):
        return self.action_space
