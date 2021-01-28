import numpy as np


class GameEmulator(object):
    """
    Main game emulator class

    Attributes:
        field_size: Field size with borders
        ratio: Empty cell to loot ratio
        current_plr_pos: Current in game player's position
        make_step: Dict with actions
        game_map: Matrix containing game map
        agent_state: Information about backpack fillness of player
    """

    def __init__(self, field_size, ratio=10, random_seed=None):
        """
        Args:
            field_size: Field size with borders
            ratio: Empty cell to loot ratio
            random_seed: numpy random seed fixation
        """
        if random_seed:
            np.random.seed(random_seed)
        self.ratio = ratio
        self.field_size = field_size
        self.current_plr_pos = (self.field_size // 2, self.field_size // 2)
        self.action_space = dict(zip(range(4), ['up', 'right', 'down', 'left']))
        self.make_step = {0: lambda x: (x[0] - 1, x[1]),  # up
                          1: lambda x: (x[0], x[1] + 1),  # right
                          2: lambda x: (x[0] + 1, x[1]),  # down
                          3: lambda x: (x[0], x[1] - 1)}  # left
        self.game_map = self._generate_char_map()
        self.agent_state = False
        self._supported_render_types = ['char', 'matrix', 'image']

    def _generate_char_map(self):
        """
        Returns:
            game_map: Matrix field_size x field_size filled by chars, where:
                l - loot,
                f - empty cell,
                e - edge cell,
                s - stock
        """
        stock_position = self.make_step[np.random.randint(4)](self.current_plr_pos)

        game_map = np.chararray((self.field_size - 2, self.field_size - 2))
        for i in range(game_map.shape[0]):
            for j in range(game_map.shape[1]):
                game_map[i, j] = np.random.choice(['f'] * self.ratio + ['l'])
            game_map[self.current_plr_pos] = 'f'
        game_map = np.pad(game_map, pad_width=1, mode='constant', constant_values='e')
        game_map[stock_position] = 's'

        return game_map

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
        return self.make_step[action](self.current_plr_pos)

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
                return self.game_map
            elif self.game_map[next_pos] in [b's']:
                self.agent_state = False
                reward = 1
        else:
            if self.game_map[next_pos] in [b'e']:
                return self.game_map
            elif self.game_map[next_pos] in [b'l']:
                self.agent_state = True
                self.game_map[next_pos] = 'f'

        self.current_plr_pos = next_pos
        return self.game_map, reward

    def render(self, render_type='char'):  # ToDo: support 'image' render type via openCV based render
        if render_type not in self._supported_render_types:
            print(f'render type {render_type} is not supported, use one of:\n{self._supported_render_types}')
            return
        if render_type == 'char':
            current_map = self.game_map.copy()
            current_map[self.current_plr_pos] = 'p'
            print(current_map)
            return current_map
        elif render_type == 'matrix':
            current_map = self.game_map.copy()
            current_map[self.current_plr_pos] = 'p'
            current_map = current_map.view(dtype=np.uint8)
            print(current_map)
            return current_map
        elif render_type == 'image':
            print('sorry, TBA))))')
            return

    def get_actions(self):
        return self.action_space


if __name__ == '__main__':
    env = GameEmulator(9, random_seed=42, ratio=3)
    env.get_actions()
    env.render('matrix')
    state, reward = env.step(0)
    state, reward = env.step(2)
    state, reward = env.step(2)
    env.render()
