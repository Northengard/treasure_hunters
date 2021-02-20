import os
import cv2
import numpy as np


class GridCell:
    """Cell class that defines each walkable Cell on the grid"""

    def __init__(self, x, y, treasure_prob):
        self.x = x
        self.y = y
        self.has_treasure = np.random.rand() < treasure_prob
        self.visited = False
        self.is_treasury = False
        self.walls = [True]*4  # Left, Right, Up, Down

    def has_children(self, grid: list) -> list:
        """Check if the Cell has any surrounding unvisited Cells that are walkable"""
        a = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        children = list()
        for x, y in a:
            if self.x + x in [len(grid), -1] or self.y + y in [-1, len(grid)]:
                continue

            child = grid[self.y + y][self.x + x]
            if child.visited:
                continue
            children.append(child)
        return children

    def erase_walls(self):
        self.walls = [False]*4

    def set_treasury(self):
        self.is_treasury = True
        self.has_treasure = False


class MapGenerator:
    def __init__(self, map_size, sparsity=2, treasure_prob=0.3, center_size=3, scale=10, random_seed=None):
        """
        map generator
        Args:
            map_size: distance from center to border of map to generate
            sparsity: sparse coefficient
            treasure_prob: probability of treasure appearance at each walkable cell
            center_size: size of max clear center area
            scale: map scale_to_visualise
        """
        self.random_seed = random_seed
        if self.random_seed:
            np.random.seed(self.random_seed)

        self.scale_percent = scale
        self._center_size = center_size
        self._sparsity = sparsity
        self._map_size = map_size
        self._player_pose = np.array([map_size // 2 + 1]*2, dtype=int)
        self._wall_color = (20, 20, 20)
        self._walk_area_color = (210, 210, 210)
        self._treasure_color = (0, 255, 255)
        self._treasure_room_color = (0, 166, 255)
        self._player_color = (0, 0, 255)
        self._treasure_prob = treasure_prob
        self._grid = self._generate_maze()
        self._treasure_list = list()

    def get_color_scheme(self):
        return {'e': self._wall_color,
                'f': self._walk_area_color,
                'l': self._treasure_color,
                's': self._treasure_room_color,
                'p': self._player_color}

    @staticmethod
    def _remove_walls(current: GridCell, choice: GridCell):
        """
        Removing the wall between two Cells
        Args:
            current: GridCell
            choice: GridCell
        Returns:
            None
        """
        if choice.x > current.x:
            current.walls[1] = False
            choice.walls[0] = False
        elif choice.x < current.x:
            current.walls[0] = False
            choice.walls[1] = False
        elif choice.y > current.y:
            current.walls[3] = False
            choice.walls[2] = False
        elif choice.y < current.y:
            current.walls[2] = False
            choice.walls[3] = False

    def _draw_walls(self, draw_grid):
        """
        Draw existing walls around Cells
        Args:
            draw_grid: list, grid where to draw
        Returns:
            grid with walls
        """
        for yi, y in enumerate(self._grid):
            for xi, x in enumerate(y):
                for i, w in enumerate(x.walls):
                    if i == 0 and w:
                        draw_grid[yi * 2 + 1][xi * 2] = self._wall_color
                    if i == 1 and w:
                        draw_grid[yi * 2 + 1][xi * 2 + 2] = self._wall_color
                    if i == 2 and w:
                        draw_grid[yi * 2][xi * 2 + 1] = self._wall_color
                    if i == 3 and w:
                        draw_grid[yi * 2 + 2][xi * 2 + 1] = self._wall_color
        return draw_grid

    def _draw_border(self, grid):
        """
        Draw a border around the maze
        Args:
            grid: list, grid where border required
        Returns:
            grid with border
        """
        # Left and Right border
        for i, x in enumerate(grid):
            x[0] = x[len(grid) - 1] = self._wall_color
            grid[i] = x

        # Top and Bottom border
        grid[0] = grid[len(grid) - 1] = [self._wall_color for _ in range(len(grid))]
        return grid

    def _draw_treasures(self, draw_grid):
        for row in self._grid:
            for grid_cell in row:
                if grid_cell.has_treasure:
                    draw_grid[grid_cell.x * 2 + 1][grid_cell.y * 2 + 1] = self._treasure_color
                    self._treasure_list.append([grid_cell.x * 2 + 1, grid_cell.y * 2 + 1])
        center_coord = self._map_size + 1
        draw_grid[center_coord][center_coord] = self._treasure_room_color
        return draw_grid

    def _prepare_grid(self):
        """Turn the grid into RGB values to then be turned into an image"""
        draw_grid = list()
        for x in range(len(self._grid) + len(self._grid) + 1):
            if x % 2 == 0:
                draw_grid.append([self._walk_area_color if x % 2 != 0 else self._wall_color
                                  for x in range(len(self._grid) + len(self._grid) + 1)])
            else:
                draw_grid.append([self._walk_area_color
                                  for _ in range(len(self._grid) + len(self._grid) + 1)])

        draw_grid = self._draw_walls(draw_grid)
        draw_grid = self._draw_treasures(draw_grid)
        draw_grid = self._draw_border(draw_grid)
        return draw_grid

    def _prepare_image(self, grid):
        """
        Turn the grid into a numpy array to then be resized
        Args:
            grid: list
        Returns:
            maze image, np.ndarray
        """
        grid = np.array(grid, dtype=np.uint8)

        width = int(grid.shape[1] * self.scale_percent)
        height = int(grid.shape[0] * self.scale_percent)
        grid = cv2.resize(grid, (width, height), interpolation=cv2.INTER_AREA)
        return grid

    def _generate_maze(self):
        """Generate a maze of Cell classes to then be turned into an image later"""
        grid = [[GridCell(x, y, self._treasure_prob) for x in range(self._map_size)] for y in range(self._map_size)]

        center_x = self._map_size // 2
        center_y = self._map_size // 2

        for _ in range(self._sparsity):
            current = grid[center_x][center_y]
            stack = list()
            start = True
            while len(stack) or start:
                start = False
                current.visited = True
                children = current.has_children(grid)

                if children:
                    choice = np.random.choice(children)
                    choice.visited = True

                    stack.append(current)

                    self._remove_walls(current, choice)

                    current = choice

                elif stack:
                    current = stack.pop()
            for row in grid:
                for cell in row:
                    cell.visited = False

        # edit center area
        grid[center_x][center_y].set_treasury()
        for x in range(center_x - 1, center_x + 2):
            for y in range(center_y - 1, center_y + 2):
                grid[x][y].erase_walls()
        return grid

    def get_maze(self):
        """
        get maze map method
        Returns:
            maze image
        """
        image = self._prepare_grid()
        image = self._prepare_image(image)
        return image, self._treasure_list


def save_image(image, path, name):
    """
    save maze images
    Args:
        image: img to save
        path: path to save image
        name: img name
    Returns:
        ret code
    """
    result = cv2.imwrite(os.path.join(path, f'{name}.png'), image)
    print(f'Status: {"Image successfully created!" if result else "Something went wrong"}')
    return result


if __name__ == '__main__':
    mg = MapGenerator(sparsity=2, map_size=30, treasure_prob=0.7)
    img = mg.get_maze()
    cv2.imshow('img', img)
    cv2.waitKey()
