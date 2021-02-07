import numpy as np


class Node:
    """
        A node class for A* Pathfinding
        parent is parent of the current Node
        position is current position of the Node in the maze
        g is cost from start to current Node
        h is heuristic based estimated cost for current Node to end Node
        f is total cost of present node i.e. :  f = g + h
    """

    def __init__(self, parent=None, position=None, step=None):
        self.parent = parent
        self.position = position
        self.step = step
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


# This function return the path of the search

class A_star_player():
    def __init__(self, state_matrix):
        self.state_matrix = state_matrix
        self.start_node = self.get_node(112)
        self.treasury_room_node = self.get_node(115)
        self.loot_list = self.create_loot_list()
        self.agent_state = False
        self.paths_to_loot = self.search_paths_for_every_loot()
        self.strategy = self.create_strategy()
        self.action_index = 0
        self.name = 'A_star_player'

    def create_loot_list(self):
        loots = np.where(self.state_matrix == 108)
        return np.stack(loots,axis=1).tolist()

    def get_node(self, type_node):
        start_position = np.where(self.state_matrix == type_node)
        return np.stack(start_position, axis=1).tolist()[0]

    def search_paths_for_every_loot(self):
        paths = []
        for loot in self.loot_list:
            path = self.search_new_step(self.state_matrix, 1, self.treasury_room_node, loot)
            paths.append(path)
        return paths

    def return_path(self, current_node):
        path = []
        while True:
            path.append(current_node.step)
            current_node = current_node.parent
            if current_node is None:
                break
        return path[:-1][::-1]

    def search_new_step(self, maze, cost, start, end):
        """
            Returns a list of tuples as a path from the given start to the given end in the given maze
            :param maze:
            :param cost
            :param start:
            :param end:
            :return:
        """
        # Create start and end node with initized values for g, h and f
        start_node = Node(None, tuple(start))
        start_node.g = start_node.h = start_node.f = 0
        end_node = Node(None, tuple(end))
        end_node.g = end_node.h = end_node.f = 0

        # Initialize both yet_to_visit and visited list
        # in this list we will put all node that are yet_to_visit for exploration.
        # From here we will find the lowest cost node to expand next
        yet_to_visit_list = []
        # in this list we will put all node those already explored so that we don't explore it again
        visited_list = []

        # Add the start node
        yet_to_visit_list.append(start_node)

        # Adding a stop condition. This is to avoid any infinite loop and stop
        # execution after some reasonable number of steps
        outer_iterations = 0
        max_iterations = (len(maze) // 2) ** 10

        # what squares do we search . serarch movement is left-right-top-bottom
        # (4 movements) from every positon

        move = [[-1, 0],  # go up
                [0, 1], # go right
                [1, 0],  # go down
                [0, -1],  # go left
                ]

        """
            1) We first get the current node by comparing all f cost and selecting the lowest cost node for further expansion
            2) Check max iteration reached or not . Set a message and stop execution
            3) Remove the selected node from yet_to_visit list and add this node to visited list
            4) Perofmr Goal test and return the path else perform below steps
            5) For selected node find out all children (use move to find children)
                a) get the current postion for the selected node (this becomes parent node for the children)
                b) check if a valid position exist (boundary will make few nodes invalid)
                c) if any node is a wall then ignore that
                d) add to valid children node list for the selected parent
    
                For all the children node
                    a) if child in visited list then ignore it and try next node
                    b) calculate child node g, h and f values
                    c) if child in yet_to_visit list then ignore it
                    d) else move the child to yet_to_visit list
        """
        # find maze has got how many rows and columns
        no_rows, no_columns = np.shape(maze)

        # Loop until you find the end

        while len(yet_to_visit_list) > 0:

            # Every time any node is referred from yet_to_visit list, counter of limit operation incremented
            outer_iterations += 1

            # Get the current node
            current_node = yet_to_visit_list[0]
            current_index = 0
            for index, item in enumerate(yet_to_visit_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

            # if we hit this point return the path such as it may be no solution or
            # computation cost is too high
            if outer_iterations > max_iterations:
                print("giving up on pathfinding too many iterations")
                return self.return_path(current_node)

            # Pop current node out off yet_to_visit list, add to visited list
            yet_to_visit_list.pop(current_index)
            visited_list.append(current_node)

            # test if goal is reached or not, if yes then return the path
            if current_node == end_node:
                return self.return_path(current_node)

            # Generate children from all adjacent squares
            children = []

            for step in range(len(move)):

                # Get node position
                node_position = (current_node.position[0] + move[step][0], current_node.position[1] +  move[step][1])

                # Make sure within range (check if within maze boundary)
                if (node_position[0] > (no_rows - 1) or
                        node_position[0] < 0 or
                        node_position[1] > (no_columns - 1) or
                        node_position[1] < 0):
                    continue

                # Make sure walkable terrain
                if maze[node_position[0]][node_position[1]] == 101:
                    continue

                # Create new node
                new_node = Node(current_node, node_position, step)

                # Append
                children.append(new_node)

            # Loop through children
            for child in children:

                # Child is on the visited list (search entire visited list)
                if len([visited_child for visited_child in visited_list if visited_child == child]) > 0:
                    continue

                # Create the f, g, and h values
                child.g = current_node.g + cost
                ## Heuristic costs calculated here, this is using eucledian distance
                child.h = (((child.position[0] - end_node.position[0]) ** 2) +
                           ((child.position[1] - end_node.position[1]) ** 2))

                child.f = child.g + child.h

                # Child is already in the yet_to_visit list and g cost is already lower
                if len([i for i in yet_to_visit_list if child == i and child.g > i.g]) > 0:
                    continue

                # Add the child to the yet_to_visit list
                yet_to_visit_list.append(child)

    def reverse_path(self, path):
        reversed_path = []
        for i in path:
            if i == 0:
                reversed_path.append(2)
            elif i == 1:
                reversed_path.append(3)
            elif i == 2:
                reversed_path.append(0)
            elif i == 3:
                reversed_path.append(1)
        return list(reversed(reversed_path))

    def create_strategy(self):
        len_paths = [len(i) for i in self.paths_to_loot]
        sorted_idxs = np.argsort(len_paths)
        sorted_paths = np.array(self.paths_to_loot)[sorted_idxs]

        path = self.search_new_step(self.state_matrix, 1, self.start_node,
                                    self.loot_list[sorted_idxs[0]])
        path_to_treasure_room = self.search_new_step(self.state_matrix, 1,
                                                     self.loot_list[sorted_idxs[0]],
                                                     self.treasury_room_node)
        strategy_actions = path + path_to_treasure_room
        for path in sorted_paths[1:]:
            strategy_actions += path
            strategy_actions += self.reverse_path(path)
        return strategy_actions

    def get_action(self, state_map, state_player):
        action = self.strategy[self.action_index]
        self.action_index = self.action_index + 1
        return action


# 0 - up,
# 1 - right,
# 2 - down,
# 3 - left
