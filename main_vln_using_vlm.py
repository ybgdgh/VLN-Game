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

# def generate_point_cloud(window):
def main(args, send_queue, receive_queue):
    args.exp_name = "vlobjectnav-"+ args.vln_mode

    log_dir = "{}/logs/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        filename=log_dir + "eval.log",
        level=logging.INFO)

    args.task_config = "vlobjectnav_hm3d.yaml"
    config = get_config(config_paths=["configs/"+ args.task_config])

    logging.info(args)
    # logging.info(config)

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.set_grad_enabled(False)

    config.defrost()
    config.DATASET.SPLIT = args.split
    # config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.gpu_id
    config.freeze()

    env = Env(config=config)

    follower = ShortestPathFollowerCompat(
        env._sim, 0.3, False
    )

    agent = VLMNav_Agent(args, follower)

    agg_metrics: Dict = defaultdict(float)

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
    # for count_episodes in trange(num_episodes):
    start = time.time()
    
    while count_episodes < num_episodes:
        obs = env.reset()
        agent.reset()
        print("Instrcution: ", obs["instruction"]['text'])
        image = transform_rgb_bgr(obs["rgb"])  # 224*224*3
        image_rgb = cv2.cvtColor(obs["rgb"], cv2.COLOR_BGR2RGB) 
        logging.info(obs["instruction"]['text'])
        if args.save_video:
            video_save_path = '{}/{}/episodes_video/eps_{}_vis.mp4'.format(
                args.dump_location, args.exp_name, agent.episode_n)
            frames = []

        count_steps = 0
        start_ep = time.time()
        while not env.episode_over:
            # 012345  停, 前进, 左, 右, 上, 下
            agent_state = env.sim.get_agent_state()

            # 每10个step询问一次
            use_vlm = count_steps % 10 == 0
            action = agent.act(obs, agent_state, use_vlm, send_queue, receive_queue)

            if action == None:
                continue
            obs = env.step(action)
            count_steps += 1
            
        if (
            action == 0 and 
            env.get_metrics()["spl"]
        ):
            # print("you successfully navigated to destination point")
            fail_case['success'] += 1
        else:
            # print("your navigation was not successful")
            if count_steps >= config.ENVIRONMENT.MAX_EPISODE_STEPS - 1:
                fail_case['exploration'] += 1
            elif agent.replan_count > 20:
                fail_case['collision'] += 1
            else:
                fail_case['detection'] += 1
        count_episodes += 1
        end = time.time()
        time_elapsed = time.gmtime(end - start)
        log = " ".join([
            "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
            "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
            "num timesteps {},".format(count_steps),
            "FPS {},".format(int(count_steps / (end - start_ep)))
        ]) + '\n'

        log += "Failed Case: collision/exploration/detection/success/total:"
        log += " {:.0f}/{:.0f}/{:.0f}/{:.0f}({:.0f}),".format(
            np.sum(fail_case['collision']),
            np.sum(fail_case['exploration']),
            np.sum(fail_case['detection']),
            np.sum(fail_case['success']),
            count_episodes) + '\n'
        
        metrics = env.get_metrics()
        for m, v in metrics.items():
            if isinstance(v, dict):
                for sub_m, sub_v in v.items():
                    agg_metrics[m + "/" + str(sub_m)] += sub_v
            else:
                agg_metrics[m] += v

        log += "Metrics: "
        log += ", ".join(k + ": {:.3f}".format(v / count_episodes) for k, v in agg_metrics.items()) + " ---({:.0f}/{:.0f})".format(count_episodes, num_episodes)

        print(log)
        logging.info(log)

        if args.save_video:
            imageio.mimsave(video_save_path, frames, fps=2)
            print(f"Video saved to {video_save_path}")
     
    avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}
    for stat_key in avg_metrics.keys():
        logger.info("{}: {:.3f}".format(stat_key, avg_metrics[stat_key]))

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