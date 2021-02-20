import os
from argparse import ArgumentParser
from time import perf_counter

import numpy as np

from connection import GameAPI
from emulator import GameEmulator
import agent


def parse_args(arg_list):
    # TODO: may be it should be better to store configurations in config files
    parser = ArgumentParser('treasure hunters game')
    parser.add_argument("-p", "--player", type=str, default='WavePlayer',
                        help="agent type that will play")
    parser.add_argument("-e", "--env", type=str, default='real',
                        choices=['emu', 'real'],
                        help='environment type to play: choose emulator or real (it should already work)')
    parser.add_argument("--host-addr", type=str, default="tcp://localhost:5556",
                        help='real environment addr to establish connection')
    parser.add_argument("--emu-map-size", type=int, default=20,
                        help='emu env map size')
    parser.add_argument("--emu-treasure-prob", type=float, default=0.2,
                        help='emu env treasure arrival probability')
    parser.add_argument("--emu-max-step", type=float, default=200,
                        help='emu env num steps before game over')
    parser.add_argument("--emu-map", type=str, default='maze',
                        choices=['empty', 'maze'],
                        help='emu env map generation type')
    parser.add_argument("--seed", type=int, default=42,
                        help='random seed')
    return parser.parse_args(arg_list)


if __name__ == '__main__':
    args = parse_args(os.sys.argv[1:])

    # for maze_size in range(20, 220, 20):
    maze_size = 20
    start = perf_counter()
    if args.env == 'emu':
        env = GameEmulator(field_size=args.emu_map_size,
                           treasure_prob=args.emu_treasure_prob,
                           max_game_steps=args.emu_max_step,
                           generation_method=args.emu_map,
                           random_seed=args.seed,
                           render_ratio=15)
    else:
        env = GameAPI(host_addr=args.host_addr)

    # gameApi test
    # for _ in range(200):
    #     action = np.random.randint(4)
    #     ret_values = env.step(action)
    #
    #     print(ret_values)
    #
    # env.close_connection()

    print(f'maze ini time: {perf_counter() - start}s')
    state_map, state_player = env.render('matrix'), False
    total_reward = 0
    env.render('image', visualize=False)

    if args.player in ["AStarPlayer", "WavePlayer"]:
        start = perf_counter()
        player = getattr(agent, args.player)(env.render("matrix"))
        print(f'player ini time: {perf_counter() - start}s')

    is_done = False
    while not is_done:
        action = player.get_action(state_map, state_player)
        state_map, state_player, reward, is_done = env.step(action)
        total_reward = total_reward + reward
        map_ = env.render('image', visualize=True)

    print(f"Game over: score of {player.name} : {total_reward}")
