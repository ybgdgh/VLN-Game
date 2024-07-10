#!/usr/bin/env python3

import argparse
import os
import random
import logging

from typing import Dict
import numpy as np
import torch
import cv2
from collections import defaultdict
from tqdm import tqdm, trange
import imageio
import threading
from multiprocessing import Process, Queue
# Gui
import open3d.visualization.gui as gui

import sys
sys.path.append(".")
import time

from habitat import Env, logger
from arguments import get_args
from habitat.config.default import get_config
from utils.shortest_path_follower import ShortestPathFollowerCompat
from utils.task import VLObjectNavEpisode
from utils.vis_gui import ReconstructionWindow
from agents.vlmnav_agent import VLMNav_Agent
from habitat.utils.visualizations import maps


def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

# 012345  停, 前进, 左, 右, 上, 下
key_map = {
    'w': 1,
    's': 0,
    'a': 2,
    'd': 3,
    '8': 4,
    '2': 5,
}

def main(args, send_queue, receive_queue):
    args.exp_name = "vlobjectnav-"+ args.vln_mode

    log_dir = "{}/logs/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        filename=log_dir + "eval.log",
        level=logging.INFO)

    args.task_config = "vlobjectnav_hm3d_v2_36.yaml"
    config = get_config(config_paths=["configs/"+ args.task_config])

    logging.info(args)

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.set_grad_enabled(False)

    config.defrost()
    config.DATASET.SPLIT = args.split
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.gpu_id
    config.freeze()

    env = Env(config=config)

    follower = ShortestPathFollowerCompat(
        env._sim, 0.3, False
    )

    agent = VLMNav_Agent(args, follower)

    num_episodes = len(env.episodes)
    if args.episode_count > -1:
        num_episodes = min(args.episode_count, len(env.episodes))
    print("num_episodes: ", num_episodes)

    fail_case = {}
    fail_case['collision'] = 0
    fail_case['success'] = 0
    fail_case['detection'] = 0
    fail_case['exploration'] = 0

    count_episodes = 0

    while count_episodes < num_episodes:
        obs = env.reset()
        agent.reset()
        print("Instrcution: ", obs["instruction"]['text'])
        count_steps = 0
        while not env.episode_over:
            agent_state = env.sim.get_agent_state()
            use_vlm = count_steps % 50 == 0
            action = agent.act(obs, agent_state, use_vlm, send_queue, receive_queue)

            user_input = input("Enter action (w, s, a, d, 8, 2): ")

            if user_input in key_map:
                action = key_map[user_input]
                obs = env.step(action)
                count_steps += 1
            else:
                print("Invalid input. No action excuting. Please try again.")

            if action == None:
                continue

            obs = env.step(action)
            count_steps += 1

    return


def visualization_thread(send_queue, receive_queue):
    app = gui.Application.instance
    app.initialize()
    mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))
    app_win = ReconstructionWindow(args, mono, send_queue, receive_queue)
    app.run()


if __name__ == "__main__":
    args = get_args()

    send_queue = Queue()
    receive_queue = Queue()

    if args.visualize:
        visualization = threading.Thread(target=visualization_thread, args=(send_queue, receive_queue,))
        visualization.start()
    main(args, send_queue, receive_queue)
