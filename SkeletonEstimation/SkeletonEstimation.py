# -*- coding: utf-8 -*-
"""## Pose_inference Class"""

from pathlib import Path
import subprocess
# # Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import sys
import decord
import mmcv
import torch.distributed as dist
from tqdm import tqdm



import numpy as np
import cv2
from matplotlib import pyplot as plt
class Pose():
    
    def __init__(self,path):
        try:
            import mmdet
            from mmdet.apis import inference_detector, init_detector
        except (ImportError, ModuleNotFoundError):
            raise ImportError('Failed to import `inference_detector` and '
                              '`init_detector` form `mmdet.apis`. These apis are '
                              'required in this script! ')
        
        try:
            import mmpose
            from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
        except (ImportError, ModuleNotFoundError):
            raise ImportError('Failed to import `inference_top_down_pose_model` and '
                              '`init_pose_model` form `mmpose.apis`. These apis are '
                              'required in this script! ')
        
        default_mmdet_root = osp.dirname(mmdet.__path__[0])
        default_mmpose_root = osp.dirname(mmpose.__path__[0])
        default_det_config = (
            f'{default_mmdet_root}/configs/faster_rcnn/'
            'faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py')
        default_det_ckpt = (
            'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/'
            'faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth')
        default_pose_config = (
            f'{default_mmpose_root}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/'
            'coco/hrnet_w32_coco_256x192.py')
        default_pose_ckpt = (
            'https://download.openmmlab.com/mmpose/top_down/hrnet/'
            'hrnet_w32_coco_256x192-c78dce93_20200708.pth')
        
        hmtx = np.array([[1.2639251386480665, 0.42630367753805887, -637.8591838122269], [-0.009783144034264873, 5.8614087492211295, -1882.755962273155], [-1.5257563583604367e-05, 0.002353355979102867, 1.0]])
        # upcourt : 1 downcourt : 2
        court = -1
        def extract_frame(video_path):
            vid = decord.VideoReader(video_path)
            return [x.asnumpy() for x in vid]
        
        def filter_people(pose, court, hmtx, p_pose = None):
            pose_mask = []
            ankle_bbox = []
            for i in range(0, len(pose)):
                l_ankle = pose[i]['keypoints'][15]
                r_ankle = pose[i]['keypoints'][16]
                mid_ankle = np.array([(l_ankle[0]+r_ankle[0])/2, 
                  (l_ankle[1]+r_ankle[1])/2, 1], dtype=np.float32)
                pp_homo = hmtx @ mid_ankle # mapping 
        
                pp_homo = pp_homo/pp_homo[2] # normalize
                if pp_homo[0] > 27.4 and pp_homo[0] < 327.6:
                    if ( pp_homo[1] > 60 and pp_homo[1] < 540 and court ==1 ) \
                    or ( pp_homo[1] > 540 and pp_homo[1] < 900 and court ==2 ) \
                    or (pp_homo[1] > 60 and pp_homo[1] < 900 and court==-1):
                        pose_mask.append(True)
                        ankle_bbox.append(mid_ankle[1])
                    else:
                        pose_mask.append(False)
                else:
        
                    pose_mask.append(False)
            if True in pose_mask:
                idx = [i for i in range(0, len(pose_mask)) if pose_mask[i]==True]
                pose = [pose[idx[i]] for i in range(0, len(idx))]
                
            else:
                pose = []
            return pose
        
        def detection_inference(model, frames):
            results = []
            for frame in frames:
                result = inference_detector(model, frame)
                results.append(result)
            return results
        
        def detection_inference_per_frame(model, frame):
            result = inference_detector(model, frame)
            return result
        
        def pose_inference(model, frames, det_results, p_pose = None):
            assert len(frames) == len(det_results)
            total_frames = len(frames)
            num_person = 10
            kp = np.zeros((num_person, total_frames, 17, 3), dtype=np.float32)
            p_pose = None
            p = False
            poses = []
        
            for i, (f, d) in enumerate(zip(frames, det_results)):
                # Align input format
                d = [dict(bbox=x) for x in list(d)]
                pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
                pose = filter_people(pose, court, hmtx, p_pose)
                # pose = filter_people(pose, court, hmtx)
                p_pose = pose
                poses.append(pose)
                for j, item in enumerate(pose):
                    kp[j, i] = item['keypoints']
            return kp, len(pose), poses

        rootdir = os.getcwd()
        print(rootdir)
        video_dir = path
        output_dir = path
        print(output_dir)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_dir+Path(video_dir).name[:-4]+'_skeleton.mp4', fourcc, 30.0, (1280,  720))
        frames = extract_frame(video_dir)
        det_model = init_detector(default_det_config, default_det_ckpt, 'cuda')
        assert det_model.CLASSES[0] == 'person', 'A detector trained on COCO is required'
        pose_model = init_pose_model(default_pose_config, default_pose_ckpt, 'cuda')
        det_results = detection_inference(det_model, frames)
        det_results = [x[0] for x in det_results]

        for i, res in enumerate(det_results):
            res = res[res[:, 4] >= 0.7]
            # * filter boxes with small areas
            box_areas = (res[:, 3] - res[:, 1]) * (res[:, 2] - res[:, 0])
            assert np.all(box_areas >= 0)
            res = res[box_areas >= 1600]
            det_results[i] = res
        
        pose_results, num_person, poses = pose_inference(pose_model, frames, det_results)
            
        for i in range(len(frames)):
            img = vis_pose_result(pose_model, frames[i], poses[i])
            out.write(img)
        out.release()