#!/usr/bin/env python3
import base64
import requests

import math
import time
import os

import torch
import open3d as o3d
from multiprocessing import Process, Queue

from habitat.core.simulator import Observations

from PIL import Image
import yaml
import quaternion
from yacs.config import CfgNode as CN
import logging

import numpy as np
import cv2
from utils.mapping import (
    merge_obj2_into_obj1,
    denoise_objects,
    filter_objects,
    merge_objects,
    gobs_to_detection_list,
    get_camera_K,
)
from utils.model_utils import compute_clip_features
from utils.slam_classes import DetectionList, MapObjectList
from utils.compute_similarities import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    merge_detections_to_objects,
    color_by_clip_sim,
    cal_clip_sim,
)

from utils.vis import init_vis_image, draw_line, vis_result_fast
from utils.explored_map_utils import (
    build_full_scene_pcd,
    detect_frontier,
)


import ast

from utils.chat_utils import chat_with_gpt
from utils.equ_ranking import Equilibrium_Ranking
from agents.system_prompt import Instruction_system_prompt
from agents.objnav_agent import ObjectNav_Agent


FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
UP_KEY = "q"
DOWN_KEY = "e"
FINISH = "f"


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


class VLMNav_Agent(ObjectNav_Agent):
    def __init__(self, args, follower=None) -> None:
        self.args = args
        if follower != None:
            self.follower = follower

        super().__init__(args=args, follower=follower)

        # ------------------------------------------------------------------
        ##### Initialize langauge navigation
        # ------------------------------------------------------------------
        self.candidate_num = 0
        self.candidate_objects = []
        self.chat_history_for_llm = []
        self.candidate_id = []

    def reset(self) -> None:
        super().reset()
        self.candidate_num = 0
        self.candidate_objects = []

    def act(
        self,
        observations: Observations,
        agent_state,
        use_vlm=False,
        send_queue=Queue(),
        receive_queue=Queue(),
    ):
        # ------------------------------------------------------------------
        ##### 1. 初始化 get the object name and init the visualization
        # ------------------------------------------------------------------
        if self.l_step == 0:
            self.init_sim_position = agent_state.sensor_states["depth"].position
            self.init_agent_position = agent_state.position
            self.init_sim_rotation = quaternion.as_rotation_matrix(
                agent_state.sensor_states["depth"].rotation
            )

            self.text_queries = observations["instruction"]["text"]  # 得到instruction
            self.text_queries = "table."  # debug 用

            self.all_objects = []
            self.landmark_data = []
            self.target_data = self.text_queries

            # 这是所有要找的包括 target和landmarks
            self.all_objects.append(self.target_data)
            print("self.all_objects: ", self.all_objects)

        # ------------------------------------------------------------------
        ##### 2. 检测和分割
        # ------------------------------------------------------------------
        image_rgb = observations["rgb"]
        # image_rgb = Image.open("/home/rickyyzliu/workspace/embodied-AI/habitat/characters.jpg")
        # image_rgb = np.array(image_rgb)
        depth = observations["depth"]
        image = transform_rgb_bgr(image_rgb)
        self.annotated_image = image
        # dino检测需要提前设置类别
        if self.args.detector == "dino":
            self.classes = self.all_objects
        get_results, detections, result_image = self.obj_det_seg.detect(
            image, image_rgb, self.classes
        )
        image_crops, image_feats, current_image_feats = compute_clip_features(
            image_rgb, detections, self.clip_model, self.clip_preprocess, self.device
        )
        if get_results:
            results = {
                "xyxy": detections.xyxy,
                "confidence": detections.confidence,
                "class_id": detections.class_id,
                "mask": detections.mask,
                "classes": self.classes,
                "image_crops": image_crops,
                "image_feats": image_feats,
            }
        else:
            results = None

        # ------------------------------------------------------------------
        ##### 3. object reconstruction
        # ------------------------------------------------------------------
        cfg = self.cfg
        depth = self._preprocess_depth(depth)
        camera_matrix_T = self.get_transform_matrix(agent_state)
        camera_position = camera_matrix_T[:3, 3]
        self.Open3D_traj.append(camera_matrix_T)
        self.relative_angle = round(np.arctan2(camera_matrix_T[2][0], camera_matrix_T[0][0]) * 57.29577951308232+ 180)

        # 2D detection结合depth转成3D
        # TODO 这里可能会去掉一些目标detection, 但是每个目标里存了它字自己的id
        fg_detection_list, bg_detection_list = gobs_to_detection_list(
            cfg=self.cfg,
            image=image_rgb,
            depth_array=depth,
            cam_K=self.camera_K,
            idx=self.l_step,
            gobs=results,
            trans_pose=camera_matrix_T,
            class_names=self.classes,
        )
        # 1) 如果当前map里已存在object, 进行merge
        if len(fg_detection_list) > 0 and len(self.objects) > 0:
            spatial_sim = compute_spatial_similarities(
                self.cfg, fg_detection_list, self.objects
            )
            visual_sim = compute_visual_similarities(
                self.cfg, fg_detection_list, self.objects
            )
            agg_sim = aggregate_similarities(self.cfg, spatial_sim, visual_sim)
            # Threshold sims according to cfg. Set to negative infinity if below threshold
            agg_sim[agg_sim < self.cfg.sim_threshold] = float("-inf")
            self.objects, detection_to_object = merge_detections_to_objects(
                self.cfg, fg_detection_list, self.objects, agg_sim
            )
        # 后处理
        if cfg.denoise_interval > 0 and (self.l_step + 1) % cfg.denoise_interval == 0:
            self.objects = denoise_objects(cfg, self.objects)
        if cfg.merge_interval > 0 and (self.l_step + 1) % cfg.merge_interval == 0:
            self.objects = merge_objects(cfg, self.objects)
        if cfg.filter_interval > 0 and (self.l_step + 1) % cfg.filter_interval == 0:
            self.objects = filter_objects(cfg, self.objects)
        # 2) 如果当前map为空, 直接加入detections
        if len(self.objects) == 0:
            detection_to_object = {}
            for i in range(len(fg_detection_list)):
                self.objects.append(fg_detection_list[i])
                detection_to_object[fg_detection_list[i]["mask_idx"][0]] = (len(self.objects) - 1)

        # ------------------------------------------------------------------
        ##### 4. 2D Obstacle Map
        # ------------------------------------------------------------------
        local_grid_pose = [
            camera_position[0] * 100 / self.args.map_resolution
            + int(self.origins_grid[0]),
            camera_position[2] * 100 / self.args.map_resolution
            + int(self.origins_grid[1]),
        ]
        pose_x = (
            int(local_grid_pose[0])
            if int(local_grid_pose[0]) < self.map_size - 1
            else self.map_size - 1
        )
        pose_y = (
            int(local_grid_pose[1])
            if int(local_grid_pose[1]) < self.map_size - 1
            else self.map_size - 1
        )

        # Adjust the centriod of the map when the robot move to the edge of the map
        if pose_x < 100:
            self.move_map_and_pose(shift=100, axis=0)
            pose_x += 100
        elif pose_x > self.map_size - 100:
            self.move_map_and_pose(shift=-100, axis=0)
            pose_x -= 100
        elif pose_y < 100:
            self.move_map_and_pose(shift=100, axis=1)
            pose_y += 100
        elif pose_y > self.map_size - 100:
            self.move_map_and_pose(shift=-100, axis=1)
            pose_y -= 100

        self.current_grid_pose = [pose_x, pose_y]

        # visualize trajectory
        self.visited_vis = draw_line(
            self.last_grid_pose, self.current_grid_pose, self.visited_vis
        )
        self.last_grid_pose = self.current_grid_pose

        # Collision check
        self.collision_check(camera_position)
        full_scene_pcd = build_full_scene_pcd(depth, image_rgb, self.args.hfov)

        # build 3D pc map
        full_scene_pcd.transform(camera_matrix_T)
        self.point_sum += self.remove_full_points_cell(full_scene_pcd, camera_position)

        obs_i_values, obs_j_values = self.update_map(
            full_scene_pcd, camera_position, self.args.map_height_cm / 100.0 / 2.0
        )

        # 求frontiers: frontier 长度, map, 中心点
        target_score, target_edge_map, target_point_list = detect_frontier(
            self.explored_map,
            self.obstacle_map,
            self.current_grid_pose,
            threshold_point=8,
        )

        # -----------------------------------------
        ##### 5. img similarity map 构建
        # -----------------------------------------
        image_clip_sim = cal_clip_sim(
            self.text_queries, current_image_feats, self.clip_model, self.clip_tokenizer
        )
        self.similarity_img_map[obs_i_values, obs_j_values] = (
            image_clip_sim.cpu().numpy()
        )

        # ------------------------------------------------------------------
        ##### 6. grounding, 核心是要算出一个self.candidate_objects 和 一个 candidate_id
        # 应该并不需要每一次都做grounding吧
        # ------------------------------------------------------------------
        if not send_queue.empty():
            text_input, self.is_running = send_queue.get()
            if text_input is not None:
                self.text_queries = text_input

        if len(self.candidate_objects) == 0:
            if use_vlm:
                # 已经得到了result image 对VLM进行提问
                # answer = self.ask_VLM(result_image)
                answer = "false_1"
                """
                函数会返回object的index, 或者是两种失败情况, 现在就考虑简单一点, 
                当返回了object的id, 就直接走过去
                若没有找到object, 则选择frontier
                """
                if answer == "false_1":
                    print("false_1")
                elif answer == "false_2":
                    print("false_2")
                else:
                    index = int(answer.split("_")[1])
                    print(f"index: {index}")
                    self.candidate_id = detection_to_object[index]
                    self.candidate_objects = [self.objects[self.candidate_id[0]]]
            
        # ------------------------------------------------------------------
        ##### 7. frontier selection
        # ------------------------------------------------------------------
        # 找到目标
        if len(self.candidate_objects) > 0:
            if self.found_goal == False:
                self.goal_map = np.zeros((self.local_w, self.local_h))
            self.goal_map[self.object_map_building(self.candidate_objects[0]["pcd"])] = 1  # 把目标位置的地方置1
            self.nearest_point = self.find_nearest_point_cloud(self.candidate_objects[0]["pcd"], camera_position)
            self.found_goal = True
            print("found goal!")
        # 如果没有找到目标
        if not self.found_goal:  
            # TODO ? 如果只有一个点, 直接把位置发给stg 为什么这种情况
            stg = None
            if np.sum(self.goal_map) == 1:
                f_pos = np.argwhere(self.goal_map == 1)
                stg = f_pos[0]

            self.goal_map = np.zeros((self.local_w, self.local_h))

            # 有边界中点, 目的还是把goal map一个地方置1
            if len(target_point_list) > 0: 
                self.no_frontiers_count = 0
                simi_max_score = []  # 存每个边界的相似度
                # 遍历边界中点
                for i in range(len(target_point_list)):
                    # 获取frontier位置(处理有可能在地图边缘的情况) min_x, max_x, min_y, max_y
                    fmb = self.get_frontier_boundaries(
                        (target_point_list[i][0], target_point_list[i][1]),
                        (self.local_w / 8, self.local_h / 8),
                        (self.local_w, self.local_h),
                    )
                    similarity_map = self.similarity_img_map
                    cropped_sim_map = similarity_map[fmb[0] : fmb[1], fmb[2] : fmb[3]]
                    # 边界上最大相似度存入
                    simi_max_score.append(np.max(cropped_sim_map))

                # 找到相似度最大的边界index: global_item
                global_item = 0
                if len(simi_max_score) > 0:
                    max_score = np.max(simi_max_score)
                    if max_score > 0.22:
                        global_item = np.argmax(simi_max_score)

                # TODO ?
                if (
                    np.array_equal(stg, target_point_list[global_item])
                    and target_score[global_item] < 30
                ):
                    self.curr_frontier_count += 1
                else:
                    self.curr_frontier_count = 0
                # TODO ?
                if (
                    self.curr_frontier_count > 20
                    or self.replan_count > 20
                    or self.greedy_stop_count > 20
                ):
                    self.obstacle_map[target_edge_map == global_item + 1] = 1
                    self.curr_frontier_count = 0
                    self.replan_count = 0
                    self.greedy_stop_count = 0

                self.goal_map[target_point_list[global_item][0], target_point_list[global_item][1]] = 1
            # TODO 这里还删除了一种情况, elif len(self.objects) > 0 and self.l_step > 200:
            # 如果没有边界存在, 就随机选取 
            else:
                goal_pose_x = int(np.random.rand() * self.map_size)
                goal_pose_y = int(np.random.rand() * self.map_size)
                self.goal_map[goal_pose_x, goal_pose_y] = 1

        # TODO goal 0 1 >1 意味着什么 
        if np.sum(self.goal_map) == 1:
            f_pos = np.argwhere(self.goal_map == 1)
            stg = f_pos[0]
            x = (stg[0] - int(self.origins_grid[0])) * self.args.map_resolution / 100.0
            y = camera_position[1]
            z = (stg[1] - int(self.origins_grid[1])) * self.args.map_resolution / 100.0
        elif np.sum(self.goal_map) == 0:
            self.found_goal == False
            goal_pose_x = int(np.random.rand() * self.map_size)
            goal_pose_y = int(np.random.rand() * self.map_size)
            self.goal_map[goal_pose_x, goal_pose_y] = 1
            f_pos = np.argwhere(self.goal_map == 1)
            stg = f_pos[0]
            x = (stg[0] - int(self.origins_grid[0])) * self.args.map_resolution / 100.0
            y = camera_position[1]
            z = (stg[1] - int(self.origins_grid[1])) * self.args.map_resolution / 100.0
        else:  # > 1
            x = self.nearest_point[0]
            y = self.nearest_point[1]
            z = self.nearest_point[2]

        # ----------------------------------------
        ###### 8. 路径规划并计算action
        # ----------------------------------------
        # 将目标点从open3d 坐标系转到habitat坐标系, open3d是什么坐标系
        Open3d_goal_pose = [x, y, z]
        Rx = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        R_habitat2open3d = self.init_sim_rotation @ Rx.T
        self.habitat_goal_pose = (
            np.dot(R_habitat2open3d, Open3d_goal_pose) + self.init_agent_position
        )
        habitat_final_pose = self.habitat_goal_pose.astype(np.float32)

        # 计算从当前位置到目标位置的路径
        plan_path = []
        plan_path = self.search_navigable_path(habitat_final_pose)

        # 如果找到可行的路径，将路径从 Habitat 坐标系转换回 Open3D 坐标系，并计算行动
        if len(plan_path) > 1:
            plan_path = np.dot(
                R_habitat2open3d.T, (np.array(plan_path) - self.init_agent_position).T
            ).T
            action = self.greedy_follower_act(plan_path)
        # 如果没有找到可行的路径，使用 Fast Marching Method (FMM) 计划一个路径, 并计算行动
        else:
            # plan a path by fmm
            self.stg, self.stop, plan_path = self._get_stg(
                self.obstacle_map, self.last_grid_pose, np.copy(self.goal_map)
            )
            plan_path = np.array(plan_path)
            plan_path_x = (
                (plan_path[:, 0] - int(self.origins_grid[0]))
                * self.args.map_resolution
                / 100.0
            )
            plan_path_y = plan_path[:, 0] * 0
            plan_path_z = (
                (plan_path[:, 1] - int(self.origins_grid[1]))
                * self.args.map_resolution
                / 100.0
            )
            plan_path = np.stack((plan_path_x, plan_path_y, plan_path_z), axis=-1)
            action = self.ffm_act()

        # ------------------------------------------------------------------
        ##### 9. 可视化 Send to Open3D visualization thread
        # ------------------------------------------------------------------
        vis_image = None
        if self.args.print_images or self.args.visualize:
            self.annotated_image = vis_result_fast(image, detections, self.classes, draw_bbox=True, draw_mask=False)
            vis_image = self._visualize(self.obstacle_map, self.explored_map, target_edge_map, self.goal_map)

        if self.args.visualize:
            if cfg.filter_interval > 0 and (self.l_step + 1) % cfg.filter_interval == 0:
                self.point_sum.voxel_down_sample(0.05)
            time_step_info = "not calculate time !"    
            receive_queue.put(
                [
                    image_rgb,
                    depth,
                    self.annotated_image,
                    self.objects.to_serializable(),
                    np.asarray(self.point_sum.points),
                    np.asarray(self.point_sum.colors),
                    self.Open3D_traj,
                    self.open3d_reset,
                    plan_path,
                    transform_rgb_bgr(vis_image),
                    Open3d_goal_pose,
                    time_step_info,
                    self.candidate_id,  # 这个是target object的index
                ]
            )
        self.open3d_reset = False
        self.last_action = action
        return action

    def Bbox_prompt(self, candidate_objects, candidate_landmarks):
        # Target object
        target_bbox = [
            {
                "centroid": [round(ele, 1) for ele in obj["bbox"].get_center()],
                "extent": [round(ele, 1) for ele in obj["bbox"].get_extent()],
            }
            for obj in candidate_objects
        ]

        # Landmark objects
        landmark_bbox = {}
        for k, candidate_landmark in candidate_landmarks.items():
            landmark_bbox[k] = [
                {
                    "centroid": [round(ele, 1) for ele in ldmk["bbox"].get_center()],
                    "extent": [round(ele, 1) for ele in ldmk["bbox"].get_extent()],
                }
                for ldmk in candidate_landmark
            ]

        # inference the target if all the objects are found
        evaluation = {}
        if (
            len(target_bbox) > 0
            and all(landmark_bbox[key] for key in landmark_bbox.keys()) > 0
        ):
            evaluation = {
                "Target Candidate BBox ('centroid': [cx, cy, cz], 'extent': [dx, dy, dz])": {
                    str(i): bbox for i, bbox in enumerate(target_bbox)
                },
                "Target Candidate BBox Volume (meter^3)": {
                    str(i): round(
                        bbox["extent"][0] * bbox["extent"][1] * bbox["extent"][2], 3
                    )
                    for i, bbox in enumerate(target_bbox)
                },
            }

            evaluation["Targe Candidate to nearest Landmark Distance (meter)"] = {
                str(i): {
                    key: min(
                        [
                            round(
                                math.sqrt(
                                    (bbox["centroid"][0] - landmark["centroid"][0]) ** 2
                                    # + (bbox["centroid"][1] - landmark_location_centroid[1]) ** 2
                                    + (bbox["centroid"][2] - landmark["centroid"][2])
                                    ** 2
                                ),
                                3,
                            )
                            for landmark in landmarks
                        ]
                    )
                    for key, landmarks in landmark_bbox.items()
                }
                for i, bbox in enumerate(target_bbox)
            }
            evaluation["Landmark Location: (cx, cy, cz)"] = (
                {
                    phrase: {
                        str(i): [round(ele, 1) for ele in landmark["centroid"]]
                        for i, landmark in enumerate(landmarks)
                    }
                    for phrase, landmarks in landmark_bbox.items()
                },
            )

            print(evaluation)
            return str(evaluation)
        else:
            return None

    def ask_VLM(result_image, instruction):

        instruction = "The table on the left side of the sofa."

        import io

        pil_image = Image.fromarray(np.uint8(result_image))
        # pil_image.show()

        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")

        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        api_key = os.getenv("OPENAI_API_KEY")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        # payload = {
        #     "model": "gpt-4o",  # gpt-4o gpt-4-vision-preview
        #     "messages": [
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "text", "text": "What's in this image?"},
        #                 {
        #                     "type": "image_url",
        #                     "image_url": {
        #                         "url": f"data:image/jpeg;base64,{img_str}"
        #                     },
        #                 },
        #             ],
        #         }
        #     ],
        #     "max_tokens": 8,  # 300
        # }

        payload = {
            "model": "gpt-4o",  # gpt-4o gpt-4-vision-preview
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Instruction: {instruction}. Here is an image:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                        },
                        {
                            "type": "text",
                            "text": "Which obj_i in the image corresponds to the object described in the instruction and why?",
                        },
                    ],
                }
            ],
            "max_tokens": 20,  # 修改为适当的值
        }

        response = requests.post(
            "https://gptproxy.llmpaas.woa.com/v1/chat/completions",
            headers=headers,
            json=payload,
        )

        print(response.json())
