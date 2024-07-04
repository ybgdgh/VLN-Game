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
    get_camera_K
)
from utils.model_utils import compute_clip_features
from utils.slam_classes import DetectionList, MapObjectList
from utils.compute_similarities import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    merge_detections_to_objects,
    color_by_clip_sim,
    cal_clip_sim
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


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
UP_KEY="q"
DOWN_KEY="e"
FINISH="f"


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


class VLObjectNav_Agent(ObjectNav_Agent):
    def __init__(self, args, follower=None) -> None:
        self.args = args
        if follower != None:
            self.follower = follower

        super().__init__(args=args, follower=follower)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        ##### Initialize langauge navigation
        # ------------------------------------------------------------------
        self.candidate_num = 0
        self.candidate_objects = []

        self.chat_history_for_llm = []

    def reset(self) -> None:
        super().reset()
        self.candidate_num = 0
        self.candidate_objects = []

        # ------------------------------------------------------------------

    def act(self, observations: Observations, agent_state, use_vlm = False, send_queue= Queue(), receive_queue= Queue()):
        time_step_info = 'Mapping time (s): \n'

        preprocess_s_time = time.time()
        # ------------------------------------------------------------------
        ##### 1. get the object name and init the visualization
        # ------------------------------------------------------------------
        # TODO 第一步是进行拆分
        if self.l_step == 0:
            self.init_sim_position = agent_state.sensor_states["depth"].position
            self.init_agent_position = agent_state.position
            self.init_sim_rotation = quaternion.as_rotation_matrix(agent_state.sensor_states["depth"].rotation)

            self.text_queries = observations["instruction"]['text']
            
            # TODO 需要修改
            self.text_queries = "table."

            self.all_objects = []
            self.landmark_data = []

            if self.args.vln_mode == "clip":
                self.target_data = self.text_queries
            else:
                if self.args.vln_mode == "llm":
                    self.chat_history_for_llm = []
                    self.chat_history_for_llm.append({"role": "system", "content": Instruction_system_prompt})
                    self.chat_history_for_llm.append({"role": "user", "content": self.text_queries})
                    response_message = chat_with_gpt(self.chat_history_for_llm, 2)
                    self.chat_history_for_llm.append({"role": "assistant", "content": response_message})

                elif self.args.vln_mode == "llm_game":
                    self.equ_ranking = Equilibrium_Ranking(self.text_queries)
                    response_message = self.equ_ranking.response_message

                generated_text = ast.literal_eval(response_message)
                ground_json = generated_text["command"]["args"]["ground_json"]
                self.target_data = ground_json["target"]["phrase"]  # target
                print("target_data: ", self.target_data)
                if "landmark" in ground_json:
                    for landmark in ground_json["landmark"].values():  # 可能会存在多个landmarks
                        if "phrase" in landmark:
                            self.landmark_data.append(landmark["phrase"])
                        self.all_objects.append(landmark["phrase"])
                    print("landmark_data: ", self.landmark_data)
        
            self.all_objects.append(self.target_data)
            # for key, value in self.landmark_data.items():
            #     self.all_objects.append(value["phrase"])
            print("self.all_objects: ", self.all_objects)  # 这是所有要找的包括 target和landmarks

        # ------------------------------------------------------------------
        ##### 2. Preprocess the observation
        # ------------------------------------------------------------------
        image_rgb = observations['rgb']
        # image_rgb = Image.open("/home/rickyyzliu/workspace/embodied-AI/habitat/characters.jpg")
        # image_rgb = np.array(image_rgb)

        depth = observations['depth']
        image = transform_rgb_bgr(image_rgb) 
        image_pil = Image.fromarray(image_rgb)
        self.annotated_image = image

        if self.args.detector == "dino":
            self.classes = self.all_objects  # all_objects是语言里的类别
        get_results, detections, result_image = self.obj_det_seg.detect(image, image_rgb, self.classes) 

        clip_s_time = time.time()  # 这个地方除了提目标的clip feat 还需要提整张图的feat
        image_crops, image_feats, current_image_feats = compute_clip_features(
            image_rgb, detections, self.clip_model, self.clip_preprocess, self.device)

        clip_e_time = time.time()
        # print('clip: %.3f秒'%(clip_e_time - clip_s_time))

        if get_results:
            results = {
                "xyxy": detections.xyxy,
                "confidence": detections.confidence,
                "class_id": detections.class_id,
                "mask": detections.mask,
                "classes": self.classes,
                "image_crops": image_crops,
                "image_feats": image_feats
                # "text_feats": text_feats
            }

        else:
            results = None

        preprocess_e_time = time.time()
        time_step_info += 'Preprocess time:%.3fs\n'%(preprocess_e_time - preprocess_s_time)

        # ------------------------------------------------------------------
        ##### 3. Object Set building
        # ------------------------------------------------------------------
        v_time = time.time()

        cfg = self.cfg
        depth = self._preprocess_depth(depth)

        camera_matrix_T = self.get_transform_matrix(agent_state)
        camera_position = camera_matrix_T[:3, 3]
        self.Open3D_traj.append(camera_matrix_T)
        self.relative_angle = round(np.arctan2(camera_matrix_T[2][0], camera_matrix_T[0][0])* 57.29577951308232 + 180)
        # print("self.relative_angle: ", self.relative_angle)

        # TODO 这里可能会去掉一些目标detection, 但是每个目标里存了它字自己的id
        fg_detection_list, bg_detection_list = gobs_to_detection_list(
            cfg = self.cfg,
            image = image_rgb,
            depth_array = depth,
            cam_K = self.camera_K,
            idx = self.l_step,
            gobs = results,
            trans_pose = camera_matrix_T,
            class_names = self.classes,
        )

        obj_time = time.time()
        objv_time = obj_time - v_time
        # print('build objects: %.3f秒'%objv_time)
        time_step_info += 'build objects time:%.3fs\n'%(objv_time)


        if len(fg_detection_list) > 0 and len(self.objects) > 0 :
            spatial_sim = compute_spatial_similarities(self.cfg, fg_detection_list, self.objects)
            visual_sim = compute_visual_similarities(self.cfg, fg_detection_list, self.objects)
            agg_sim = aggregate_similarities(self.cfg, spatial_sim, visual_sim)
            # Threshold sims according to cfg. Set to negative infinity if below threshold
            agg_sim[agg_sim < self.cfg.sim_threshold] = float('-inf')
            self.objects, detection_to_object = merge_detections_to_objects(self.cfg, fg_detection_list, self.objects, agg_sim)

        # Perform post-processing periodically if told so
        if cfg.denoise_interval > 0 and (self.l_step+1) % cfg.denoise_interval == 0:
            self.objects = denoise_objects(cfg, self.objects)
        if cfg.merge_interval > 0 and (self.l_step+1) % cfg.merge_interval == 0:
            self.objects = merge_objects(cfg, self.objects)
        if cfg.filter_interval > 0 and (self.l_step+1) % cfg.filter_interval == 0:
            self.objects = filter_objects(cfg, self.objects)

        sim_time = time.time()
        sim_obj_time = sim_time - obj_time 
        # print('calculate merge: %.3f秒'%sim_obj_time)
        time_step_info += 'calculate merge time:%.3fs\n'%(sim_obj_time)

        if len(self.objects) == 0:
            # Add all detections to the map
            detection_to_object = {}
            for i in range(len(fg_detection_list)):
                self.objects.append(fg_detection_list[i])
                detection_to_object[fg_detection_list[i]["mask_idx"][0]] = len(self.objects)-1

        # ------------------------------------------------------------------
        ##### 4. 2D Obstacle Map
        # ------------------------------------------------------------------
        f_map_time = time.time()

        local_grid_pose = [camera_position[0]*100/self.args.map_resolution + int(self.origins_grid[0]), 
                      camera_position[2]*100/self.args.map_resolution + int(self.origins_grid[1])]
        pose_x = int(local_grid_pose[0]) if int(local_grid_pose[0]) < self.map_size-1 else self.map_size-1
        pose_y = int(local_grid_pose[1]) if int(local_grid_pose[1]) < self.map_size-1 else self.map_size-1

        # Adjust the centriod of the map when the robot move to the edge of the map
        if pose_x < 100:
            self.move_map_and_pose(shift = 100, axis=0)
            pose_x += 100
        elif pose_x > self.map_size - 100:
            self.move_map_and_pose(shift = -100, axis=0)
            pose_x -= 100
        elif pose_y < 100:
            self.move_map_and_pose(shift = 100, axis=1)
            pose_y += 100
        elif pose_y > self.map_size - 100:
            self.move_map_and_pose(shift = -100, axis=1)
            pose_y -= 100

        self.current_grid_pose = [pose_x, pose_y]

        # visualize trajectory
        self.visited_vis = draw_line(self.last_grid_pose, self.current_grid_pose, self.visited_vis)
        self.last_grid_pose = self.current_grid_pose

        # Collision check
        self.collision_check(camera_position)
        full_scene_pcd = build_full_scene_pcd(depth, image_rgb, self.args.hfov)

        # build 3D pc map
        full_scene_pcd.transform(camera_matrix_T)
        self.point_sum += self.remove_full_points_cell(full_scene_pcd, camera_position)

        obs_i_values, obs_j_values = self.update_map(full_scene_pcd, camera_position, self.args.map_height_cm / 100.0 /2.0)

        target_score, target_edge_map, target_point_list = detect_frontier(self.explored_map, self.obstacle_map, self.current_grid_pose, threshold_point=8)

        v_map_time = time.time()
        # print('voxel map: %.3f秒'%(v_map_time - f_map_time))

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        ##### Calculate similarities for objects and landmarks
        # ------------------------------------------------------------------
        # text_queries = ['sofa chair']
        if not send_queue.empty():
            text_input, self.is_running = send_queue.get()
            if text_input is not None:
                self.text_queries = text_input
            # print("self.text_queries: ", self.text_queries)

        clip_time = time.time()

        total_candidate_objects = []  # 这里既包括target 也包括landmarks
        total_similarities = []
        candidate_target = []
        candidate_landmarks = {}
        candidate_id = []
        similarity_threshold = 0.29
        similarities = None

        #  ----------------------------------------
        # 5. 进行grounding
        # -----------------------------------------
        '''
        grounding的部分:
        首先根据相似度去grounding 3D target objects
        然后看语言里是否包含landmarks的内容, 不包括则直接确定target object
        若语言里有landmarks, 则利用相似度grounding 3D landmarks, 然后通过对LLM提问确定最终的target object
        构建obj similarity map
        构建img similarity map
        '''
        if len(self.objects) > 0:  
            # 1. 从map 中找到语言里包括的target object(根据相似性): candidate_target(3D)
            self.objects, target_similarities = color_by_clip_sim("looks like a " + self.target_data, # 查询的文本
                                                        self.objects,  # 3D 对象列表
                                                        self.clip_model, 
                                                        self.clip_tokenizer)

            # candidate条件: >阈值, 多次(2)看到, 或这个检测目标的检测阈值>0.75(因为检测就是检测语言里的目标)
            candidate_target = [self.objects[i] for i in range(len(self.objects)) 
                                if (target_similarities[i] > similarity_threshold and \
                                    self.objects[i]['num_detections'] > 2) \
                                    or \
                                    max(self.objects[i]['conf']) > 0.75] 

            candidate_id = [i for i in range(len(self.objects)) 
                                if (target_similarities[i] > similarity_threshold and \
                                    self.objects[i]['num_detections'] > 2) \
                                    or \
                                    max(self.objects[i]['conf']) > 0.75]

            if len(candidate_target) > 0:
                print("find targets: " + self.target_data, len(candidate_target), " ", max(target_similarities.cpu().numpy()))

            total_candidate_objects.extend(candidate_target)
            total_similarities.append(target_similarities)
            print("max_sim target: ", max(target_similarities.cpu().numpy()))

            # 2. 综合landmarks确定最终的target object 3D
            # 2.1 当语言里不存在landmark, 则 self.candidate_objects就是相似度求出的target object
            if len(self.landmark_data) == 0:
                self.candidate_objects = candidate_target
            else:
                # 2.2 当语言里存在landmark的情况
                # 2.2.1 先从map中找到语言里包括的landmarks(根据相似度): candidate_landmarks["text label"] (3D)
                for landmark in self.landmark_data:
                    self.objects, landmark_similarities = color_by_clip_sim("looks like a " + landmark, 
                                                                            self.objects, 
                                                                            self.clip_model, 
                                                                            self.clip_tokenizer, 
                                                                            color_set = False)

                    candidate_landmarks[landmark] = [self.objects[i] for i in range(len(self.objects)) 
                                    if landmark_similarities[i] > similarity_threshold]
                    candidate_id.extend([i for i in range(len(self.objects)) 
                                    if landmark_similarities[i] > similarity_threshold])

                    # if len(candidate_landmarks[landmark]) > 0:
                    print("find ", landmark , ": ", len(candidate_landmarks[landmark]), " ", max(landmark_similarities.cpu().numpy()))
                    total_candidate_objects.extend(candidate_landmarks.values())
                    total_similarities.append(landmark_similarities)

                # 2.2.2 对LLM进行提问, 让其返回target id, 确定最终的3D object
                # found all objects after filting and candidate objects increase
                if (
                    len(candidate_target) > 0
                    and all(candidate_landmarks[key] for key in candidate_landmarks.keys())> 0  # key就是语言里的label 
                    and (self.l_step + 1) % cfg.filter_interval == 0  # 时间上 
                    and len(total_candidate_objects) > self.candidate_num  # 总的candidatelabel
                ):
                    user_prompt = self.Bbox_prompt(
                        candidate_target, candidate_landmarks
                    )

                    if user_prompt != None:
                        if self.args.vln_mode == "llm":
                            self.chat_history_for_llm.append({"role": "user", "content": user_prompt})
                            response_message = chat_with_gpt(self.chat_history_for_llm)
                            self.chat_history_for_llm.append({"role": "assistant", "content": response_message})

                            ground_json = ast.literal_eval(response_message)

                            # 如果完成了grounding
                            if ground_json["command"]["name"] == "finish_grounding":
                                # TODO self.candidate_objects 就是找到的3D 目标, LLM回答他的id
                                self.candidate_objects = [
                                    candidate_target[
                                        int(
                                            ground_json["command"]["args"][
                                                "top_1_object_id"
                                            ]
                                        )
                                    ]
                                ]
                                print("Found candidate_objects!")

                        elif self.args.vln_mode == "llm_game":
                            candidate_index = self.equ_ranking.equilibrium_search(user_prompt)
                            if candidate_index == len(candidate_target) : # -1: not sure
                                print("reject all candidates")
                            else:
                                self.candidate_objects = [candidate_target[candidate_index]]
                                print("Found candidate_objects!")  

                    self.candidate_num = len(total_candidate_objects)

            # 3. obj similarity map 构建
            similarities, _ = torch.max(torch.stack(total_similarities), dim = 0)
            similarities = similarities.cpu().numpy()
            for i in range(len(self.objects)):
                if similarities[i] > 0.24:
                    # TODO 计算 obj similarity map
                    self.similarity_obj_map[self.object_map_building(self.objects[i]['pcd'])] = similarities[i]  

        # 应该以一定的频率执行
        if False:
            # 已经得到了result image 对VLM进行提问
            answer = self.ask_VLM(result_image)
            '''
            函数会返回object的index, 或者是两种失败情况, 现在就考虑简单一点, 
            当返回了object的id, 就直接走过去
            若没有找到object, 则选择frontier
            '''
            if answer == "false_1":
                print("false_1")
            elif answer == "false_2":
                print("false_2")
            else:
                index = int(answer.split("_")[1])
                print(f"index: {index}")
        
        if use_vlm:
            # if len(fg_detection_list) != len(results["mask"]):
            #     raise ValueError("fg lens != all detections")

            # 遍历前景目标, 根据detection id 确定object id , 可视化object id check对不对
            for i in range(len(fg_detection_list)):
                object_index = detection_to_object[fg_detection_list[i]["mask_idx"][0]]
                object_map = self.objects[object_index]
                # o3d.visualization.draw_geometries([object_map["pcd"]])  # TODO 为什么这里的目标点云的颜色被改变了
                print(1)

        # 4. img similarity map 构建
        image_clip_sim = cal_clip_sim(self.text_queries, current_image_feats, self.clip_model, self.clip_tokenizer)
        self.similarity_img_map[obs_i_values, obs_j_values] = image_clip_sim.cpu().numpy()  

        f_clip_time = time.time()
        # print('calculate clip sim: %.3f秒'%(f_clip_time - clip_time))
        time_step_info += 'calculate clip sim time:%.3fs\n'%(f_clip_time - clip_time)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        ##### frontier selection
        # ------------------------------------------------------------------

        if len(self.candidate_objects) > 0 : 
            if self.found_goal == False:
                self.goal_map = np.zeros((self.local_w, self.local_h))
            self.goal_map[self.object_map_building(self.candidate_objects[0]['pcd'])] = 1
            self.nearest_point = self.find_nearest_point_cloud(self.candidate_objects[0]['pcd'], camera_position)
            self.found_goal = True
            print("found goal!")
        # elif len(self.objects) > 0 and max(similarities) < similarity_threshold:
        #     self.found_goal = False

        if not self.found_goal:
            stg = None
            if np.sum(self.goal_map) == 1:
                f_pos = np.argwhere(self.goal_map == 1)
                stg = f_pos[0]

            self.goal_map = np.zeros((self.local_w, self.local_h))
            if len(target_point_list) > 0:  # TODO 有边界中点
                self.no_frontiers_count = 0
                simi_max_score = []
                for i in range(len(target_point_list)):
                    fmb = self.get_frontier_boundaries((target_point_list[i][0], 
                                                    target_point_list[i][1]),
                                                    (self.local_w/8, self.local_h/8),
                                                    (self.local_w, self.local_h))
                    similarity_map = np.max(np.stack([self.similarity_obj_map, self.similarity_img_map]), axis=0)
                    cropped_sim_map = similarity_map[fmb[0]:fmb[1], fmb[2]:fmb[3]]
                    simi_max_score.append(np.max(cropped_sim_map))

                # print("simi_max_score: ", simi_max_score)
                global_item = 0
                if len(simi_max_score) > 0:
                    # print(simi_max_score)
                    if max(simi_max_score) > 0.22:
                        global_item = simi_max_score.index(max(simi_max_score))

                if np.array_equal(stg, target_point_list[global_item]) and target_score[global_item] < 30:
                    self.curr_frontier_count += 1
                else:
                    self.curr_frontier_count = 0

                if self.curr_frontier_count > 20 or self.replan_count > 20 or self.greedy_stop_count > 20:
                    self.obstacle_map[target_edge_map == global_item+1] = 1
                    self.curr_frontier_count = 0
                    self.replan_count = 0
                    self.greedy_stop_count = 0

                self.goal_map[target_point_list[global_item][0], target_point_list[global_item][1]] = 1
            elif len(self.objects) > 0 and self.l_step > 200:
                self.no_frontiers_count += 1
                self.goal_map = np.zeros((self.local_w, self.local_h))
                max_index = np.argmax(similarities)
                self.goal_map[self.object_map_building(self.objects[max_index]['pcd'])] = 1
                self.nearest_point = self.find_nearest_point_cloud(self.objects[max_index]['pcd'], camera_position)

                if self.no_frontiers_count > 5 and max(similarities) > 0.25 :
                    self.found_goal = True

            else:
                goal_pose_x = int(np.random.rand() * self.map_size)
                goal_pose_y = int(np.random.rand() * self.map_size)
                self.goal_map[goal_pose_x, goal_pose_y] = 1

        if np.sum(self.goal_map) == 1:
            f_pos = np.argwhere(self.goal_map == 1)
            stg = f_pos[0]
            x = (stg[0] - int(self.origins_grid[0])) * self.args.map_resolution / 100.0
            y = camera_position[1]
            z = (stg[1] - int(self.origins_grid[1])) * self.args.map_resolution / 100.0
        elif np.sum(self.goal_map) == 0:
            # stg = self._get_closed_goal(self.obstacle_map, self.last_grid_pose, find_big_connect(self.goal_map))
            self.found_goal == False
            goal_pose_x = int(np.random.rand() * self.map_size)
            goal_pose_y = int(np.random.rand() * self.map_size)
            self.goal_map[goal_pose_x, goal_pose_y] = 1
            f_pos = np.argwhere(self.goal_map == 1)
            stg = f_pos[0]

            x = (stg[0] - int(self.origins_grid[0])) * self.args.map_resolution / 100.0
            y = camera_position[1]
            z = (stg[1] - int(self.origins_grid[1])) * self.args.map_resolution / 100.0

        else: # > 1           
            x = self.nearest_point[0]
            y = self.nearest_point[1]
            z = self.nearest_point[2]

        Open3d_goal_pose = [x, y, z]

        Rx = np.array([[0, 0, -1],
                    [0, 1, 0],
                    [1, 0, 0]])
        R_habitat2open3d = self.init_sim_rotation @ Rx.T
        self.habitat_goal_pose = np.dot(R_habitat2open3d, Open3d_goal_pose) + self.init_agent_position
        habitat_final_pose = self.habitat_goal_pose.astype(np.float32)

        plan_path = []
        plan_path = self.search_navigable_path(
            habitat_final_pose
        )

        if len(plan_path) > 1:
            plan_path = np.dot(R_habitat2open3d.T, (np.array(plan_path) - self.init_agent_position).T).T
            action = self.greedy_follower_act(plan_path)

        else:
            # plan a path by fmm
            self.stg, self.stop, plan_path = self._get_stg(self.obstacle_map, self.last_grid_pose, np.copy(self.goal_map))
            plan_path = np.array(plan_path) 
            plan_path_x = (plan_path[:, 0] - int(self.origins_grid[0])) * self.args.map_resolution / 100.0
            plan_path_y = plan_path[:, 0] * 0
            plan_path_z = (plan_path[:, 1] - int(self.origins_grid[1])) * self.args.map_resolution / 100.0

            plan_path = np.stack((plan_path_x, plan_path_y, plan_path_z), axis=-1)

            action = self.ffm_act()

        # action = self.keyboard_act()

        vis_image = None
        if self.args.print_images or self.args.visualize:
            self.annotated_image  = vis_result_fast(image, detections, self.classes, draw_bbox=True, draw_mask=False)
            vis_image = self._visualize(self.obstacle_map, self.explored_map, target_edge_map, self.goal_map)

        if len(self.objects) > 0:
            time_step_info += 'max similarity: %3fs\n'%(np.max(similarities))

        # ------------------------------------------------------------------
        ##### Send to Open3D visualization thread
        # ------------------------------------------------------------------
        if self.args.visualize:
            if cfg.filter_interval > 0 and (self.l_step+1) % cfg.filter_interval == 0:
                self.point_sum.voxel_down_sample(0.05)
            receive_queue.put([image_rgb, 
                                depth, 
                                self.annotated_image , 
                                self.objects.to_serializable(), 
                                np.asarray(self.point_sum.points), 
                                np.asarray(self.point_sum.colors), 
                                self.Open3D_traj,
                                self.open3d_reset,
                                plan_path,
                                transform_rgb_bgr(vis_image),
                                Open3d_goal_pose,
                                time_step_info,
                                candidate_id]
                                )   
        self.open3d_reset = False

        self.last_action = action

        # transfer_time = time.time()
        # time_step_info += 'transfer data time:%.3fs\n'%(transfer_time - dd_map_time)

        # print(time_step_info)

        return action

    def Bbox_prompt(self, candidate_objects, candidate_landmarks):
        # Target object
        target_bbox = [{"centroid": [round(ele, 1) for ele in obj['bbox'].get_center()], 
                        "extent": [round(ele, 1) for ele in obj['bbox'].get_extent()]} 
                        for obj in candidate_objects ]

        # Landmark objects
        landmark_bbox = {}
        for k,candidate_landmark in candidate_landmarks.items():
            landmark_bbox[k] = [{"centroid": [round(ele, 1) for ele in ldmk['bbox'].get_center()], 
                            "extent": [round(ele, 1) for ele in ldmk['bbox'].get_extent()]}  
                            for ldmk in candidate_landmark]

        # inference the target if all the objects are found
        evaluation = {}
        if len(target_bbox) > 0 and all(landmark_bbox[key] for key in landmark_bbox.keys()) > 0:
            evaluation = {
                "Target Candidate BBox ('centroid': [cx, cy, cz], 'extent': [dx, dy, dz])": {
                    str(i): bbox for i, bbox in enumerate(target_bbox)
                },
                "Target Candidate BBox Volume (meter^3)": {
                    str(i): round(bbox["extent"][0] * bbox["extent"][1] * bbox["extent"][2], 3)
                    for i, bbox in enumerate(target_bbox)
                },
            }

            evaluation["Targe Candidate to nearest Landmark Distance (meter)"] = {
                str(i): { key: min([round(
                    math.sqrt(
                        (bbox["centroid"][0] - landmark["centroid"][0]) ** 2
                        # + (bbox["centroid"][1] - landmark_location_centroid[1]) ** 2
                        + (bbox["centroid"][2] - landmark["centroid"][2]) ** 2
                    ),
                    3,
                ) 
                    for landmark in landmarks])
                    for key, landmarks in landmark_bbox.items()}
                for i, bbox in enumerate(target_bbox)
            }
            evaluation["Landmark Location: (cx, cy, cz)"] = {
                    phrase: {str(i): [round(ele, 1) for ele in landmark["centroid"]]
                    for i, landmark in enumerate(landmarks)}
                    for phrase, landmarks in landmark_bbox.items()
                },

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


        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

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
                        {"type": "text", "text": f"Instruction: {instruction}. Here is an image:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_str}"
                            },
                        },
                        {"type": "text", "text": "Which obj_i in the image corresponds to the object described in the instruction and why?"}
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


