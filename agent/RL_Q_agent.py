import numpy as np
from collections import deque
import random
import os
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from emulator import GameEmulator
from scipy.spatial import distance

DISCOUNT = 1
REPLAY_MEMORY_SIZE = 1000
MIN_REPLAY_MEMORY_SIZE = 100
MINIBATCH_SIZE = 64
MODEL_NAME = 'Model'

EPISODES = 4000

EPSILON_DECAY = 0.995
MIN_EPSILON = 0.001

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)


def generate_session(agent, max_iterations=1000, visualize=False):
    agent.eval()
    states, actions, agent_states = [], [], []
    total_reward = 0
    s, agent_state, is_done = env.reset()
    with torch.no_grad():
        for i in range(max_iterations):

            input = torch.Tensor([s]).unsqueeze(1)
            input = input.to("cuda:0")

            out = agent(input, [agent_state])
            a = np.argmax(out.cpu().detach().numpy())
            action = a

            new_s, agent_state, r, is_done = env.step(action)
            if visualize:
                print(i, agent_state, total_reward)
                env.render('image', visualize=True)
            states.append(s)
            agent_states.append(agent_state)
            actions.append(action)
            total_reward += r
            s = new_s
            if is_done:
                break
    return states, agent_states, actions, total_reward


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(16)

        # self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(w)
        convh = conv2d_size_out(h)
        linear_input_size = convw * convh * 16
        self.head = nn.Linear(linear_input_size + 1, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, agent_state):
        x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        a_state = torch.Tensor(agent_state).unsqueeze(1).to("cuda:0")
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, a_state), dim=1)
        return self.head(x)


class DQNAgent:
    def __init__(self, size_w, size_h, n_actions):

        self.model = self.create_model(size_w, size_h, n_actions)

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def create_model(self, size_h, size_w, n_actions):
        model = DQN(size_w, size_h, n_actions)
        model = model.to("cuda:0")
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):

        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])
        curremt_agent_states = np.array([transition[1] for transition in minibatch])
        input = torch.Tensor(current_states).unsqueeze(1)
        input = input.to("cuda:0")
        current_qs_list = self.model(input, curremt_agent_states).cpu().detach().numpy()

        new_current_states = np.array([transition[4] for transition in minibatch])
        new_current_agent_states = np.array([transition[5] for transition in minibatch])

        input = torch.Tensor(new_current_states).unsqueeze(1)
        input = input.to("cuda:0")
        future_qs_list = self.model(input, new_current_agent_states).cpu().detach().numpy()

        X = []
        X_agent_state = []
        y = []

        for index, (current_state, curremt_agent_states, action, reward, new_current_state, new_current_agent_states,
                    done) in enumerate(minibatch):

            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            X_agent_state.append(curremt_agent_states)
            y.append(current_qs)

        self.model.train()

        target = torch.tensor([y])
        target = target.to("cuda:0")
        self.optimizer.zero_grad()

        input = torch.Tensor(X).unsqueeze(1)
        input = input.to("cuda:0")

        out = self.model(input, X_agent_state)
        loss = self.criterion(out, target)
        loss.backward()
        self.optimizer.step()

    def get_qs(self, state, agent_state):
        input = torch.Tensor([state]).unsqueeze(1)
        input = input.to("cuda:0")

        return self.model(input, [agent_state]).cpu().detach().numpy()


def get_node(maze, type_node):
    start_position = np.where(maze == type_node)
    return np.stack(start_position, axis=1).tolist()[0]


def create_loot_list(maze):
    loots = np.where(maze == 108)
    return np.stack(loots, axis=1).tolist()


def generate_reward_map(state, stock_place):
    reward_map = np.zeros((len(state), len(state)))
    for i in range(len(state)):
        for j in range(len(state)):
            reward_map[i][j] = distance.euclidean([i, j], stock_place)
    return reward_map


if __name__ == "__main__":
    size = 6
    env = GameEmulator(field_size=size,
                       treasure_prob=0.2,
                       max_game_steps=50,
                       generation_method="empty",
                       random_seed=42,  # np.random.randint(1, 1000),
                       render_ratio=15)
    env.render('image', visualize=True)

    n_actions = 4

    epsilon = 1

    last_average_reward = 0
    ep_rewards = [-200]

    agent = DQNAgent(size + 1, size + 1, n_actions)

    if not os.path.isdir('models'):
        os.makedirs('models')

    best_reward = -200

    ep_total_reward = []
    for episode in range(1, EPISODES + 1):

        episode_reward = 0

        current_state, current_agent_state, done = env.reset()

        done = False
        while not done:

            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(current_state, current_agent_state))
            else:
                action = np.random.randint(0, n_actions)

            new_state, new_agent_state, reward, done = env.step(action)
            treasure_list = env._treasure_list

            if new_agent_state:
                reward_map = generate_reward_map(new_state, get_node(new_state, 115))
                new_player_pose = get_node(new_state, 112)
                reward = reward - reward_map[new_player_pose[0]][new_player_pose[1]] / len(new_state)
            if current_agent_state and not new_agent_state:
                reward = reward + 10
            if np.array_equal(new_state, current_state) and new_agent_state:
                reward_map = generate_reward_map(new_state, get_node(new_state, 115))
                new_player_pose = get_node(new_state, 112)
                reward = reward - reward_map[new_player_pose[0]][new_player_pose[1]] / len(new_state)

            # current_player_pose = get_node(current_state, 112)
            # if current_agent_state:
            #     current_goal_node = get_node(current_state, 115)
            #     current_dist_goal = distance.euclidean(current_player_pose, current_goal_node)
            # else:
            #     loot_lists = create_loot_list(current_state)
            #     current_dist_goal = np.inf
            #     for l in loot_lists:
            #         dist = distance.euclidean(current_player_pose, l)
            #         if dist < current_dist_goal:
            #             current_dist_goal = dist
            #
            # new_player_pose = get_node(new_state, 112)
            # if new_agent_state:
            #     new_goal_node = get_node(new_state, 115)
            #     new_dist_goal = distance.euclidean(new_player_pose, new_goal_node)
            # else:
            #     loot_lists = create_loot_list(new_state)
            #     new_dist_goal = np.inf
            #     for l in loot_lists:
            #         dist = distance.euclidean(new_player_pose, l)
            #         if dist < new_dist_goal:
            #             new_dist_goal = dist
            #
            # if new_dist_goal > current_dist_goal:
            #     reward = reward -1

            episode_reward += reward

            agent.update_replay_memory(
                (current_state, current_agent_state, action, reward, new_state, new_agent_state, done))

            current_state = new_state
            current_agent_state = new_agent_state
        agent.train()

        ep_rewards.append(episode_reward)

        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
        ep_tmp_reward = []
        for i in range(10):
            _, __, ___, tmp_reward = generate_session(agent.model, visualize=False)
            ep_tmp_reward.append(tmp_reward)
        tmp_reward = np.mean(ep_tmp_reward)
        ep_total_reward.append(tmp_reward)
        print("Iteration - ", episode, "current reward - ", ep_rewards[-1], "Validation reward - ", tmp_reward)
        if tmp_reward >= last_average_reward:
            torch.save(agent.model, f'best_model_dql.model')
            last_average_reward = tmp_reward

        if tmp_reward >= 900:
            break

        if episode % 200 == 0:
            _, __, ___, tmp_reward = generate_session(agent.model, visualize=True)

    plt.plot(ep_total_reward)
    plt.plot(ep_rewards)
    plt.show()

    agent.model = torch.load(f'best_model_dql.model')
    agent.model.eval()

    env = GameEmulator(field_size=size,
                       treasure_prob=0.2,
                       max_game_steps=200,
                       generation_method="maze",
                       random_seed=42,
                       render_ratio=15)
    while True:
        state = env.reset()
        done = False
        while not done:
            out = agent.model(torch.Tensor(state))
            action = np.argmax(out.detach().numpy())
            new_s, __, done, _ = env.step(action)
            env.render()
            state = new_s
