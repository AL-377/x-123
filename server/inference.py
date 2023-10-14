#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：x-123
@File    ：inference.py
@Author  ：Aidan Lew
@Date    ：2023/10/6 8:59
"""
from functools import partial
from pathlib import Path
from typing import Tuple
import time
import os

import numpy as np
import cv2
from server.utils.image import pad_resize_image, scale_coords, draw_bbox_on_image
from deepface import DeepFace


backends = [
    'opencv',
    'ssd',
    'dlib',
    'mtcnn',
    'retinaface',
    'mediapipe',
    'yolov8',
    'yunet',
]


def preprocess(img: np.ndarray, width: int = 448, height: int = 448, new_type: type = np.float32) -> np.ndarray:
    """
    Preprocess input cv2 image
    Args:
        img: cv2 input image
        width: model input width
        height: model input height
        new_type: type to which the image is converted to
    """
    width = 448 if width is None else width
    height = 448 if height is None else height
    resized = pad_resize_image(img, (width, height))
    img_in = np.transpose(resized, (2, 0, 1)).astype(new_type)  # HWC -> CHW
    return img_in


def postprocess(results, orig_img_size: Tuple[int, int], input_img_size: Tuple[int, int]) -> dict:
    """
    Postprocess detections & face features from triton-server
    Args:
        results: triton server inference result
        orig_img_size: (width, height)
        input_img_size: (width, height)
    """
    predictions = {"face_feats": [], "face_detections": []}

    results_arr = results.as_numpy("ENSEMBLE_FACE_FEAT")
    if results_arr.any():
        for i, result in enumerate(results_arr):
            # result = result
            # pool
            result = result.squeeze()
            predictions["face_feats"].append(result)

    box_arr = results.as_numpy("ENSEMBLE_FACE_DETECTOR_BBOXES").copy()
    conf_arr = results.as_numpy("ENSEMBLE_FACE_DETECTOR_CONFS").tolist()

    if box_arr is not None and box_arr.any():
        model_w, model_h = input_img_size
        orig_w, orig_h = orig_img_size
        box_arr *= [model_w, model_h, model_w, model_h]
        box_arr = scale_coords((model_h, model_w), box_arr, (orig_h, orig_w))
        for i, result in enumerate(box_arr):
            result = result.copy()
            x_min, y_min, x_max, y_max = result.astype(int).tolist()
            conf = conf_arr[i]
            predictions["face_detections"].append(
                {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max, "confidence": conf})
    return predictions


def run_inference(face_path, face_feat_model, face_det_thres,
                  face_count_thres = 1, mode="register"):
    # res format
    res = {}
    # detect and extract
    try:
        face_objs = DeepFace.extract_faces(
            img_path=face_path,
            target_size=(224, 224),
            detector_backend=backends[4]
        )
    except Exception:
        print("fail to detect face in img")
        res["status"] = 0
        res["face_detections"] = False
    else:
        # face_objs format as [{},{}]
        filtered_faces_info = [d for d in face_objs if d["confidence"] > face_det_thres]
        filtered_faces_info = sorted(filtered_faces_info,key=lambda x:x["confidence"])
        if filtered_faces_info[0]["confidence"] < face_det_thres:
            print("fail to detect face in img")
            res["status"] = 1
            res["face_detections"] = False
        elif mode == "register":
            try:
                # extract only the feature of first face(top conf)
                # skip the pre-process process
                embedding_objs = DeepFace.represent(
                    img_path=filtered_faces_info[0]["face"],
                    model_name=face_feat_model,
                    detector_backend="skip"
                )
                # embedding_objs is all face res,so [res]
                embedding = embedding_objs[0]["embedding"]
            except Exception:
                print("fail to get the embedding of face")
                res["status"] = 0
                res["face_detections"] = False
            else:
                res["status"] = 1
                res["face_detections"] = True
                res["face_feats"] = [embedding]
                res["face_boxes"] = [filtered_faces_info[0]["facial_area"]]
        elif mode == "recognize":
            # store all res
            res["face_feats"] = []
            res["face_boxes"] = []
            # face vector separated
            for i in range(len(filtered_faces_info)):
                embedding_objs = DeepFace.represent(
                    img_path=filtered_faces_info[i]["face"],
                    model_name=face_feat_model,
                    detector_backend="skip"
                )
                embedding = embedding_objs[0]["embedding"]
                res["face_feats"].append(embedding)
                res["face_boxes"].append(filtered_faces_info[i]["facial_area"])
            res["status"] = 1
            res["face_detections"] = True

    finally:
        return res
