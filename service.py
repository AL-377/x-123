#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：x-123 
@File    ：service.py
@Author  ：Aidan Lew
@Date    ：2023/10/6 7:47 
"""
import pymysql
from pymilvus import MilvusException

from dao.milvus import get_milvus_collec_conn
from dao.mysql import (insert_person_data_into_sql,
                       select_all_person_data_from_sql,
                       select_person_data_from_sql_with_id,
                       delete_person_data_from_sql_with_id,
                       update_avatar_url_in_sql,
                       select_person_avatar_from_sql_with_id)
from pymysql.cursors import DictCursor

from config import (MYSQL_HOST, MYSQL_PORT,
                    MYSQL_USER, MYSQL_PASSWORD,
                    MYSQL_DATABASE, MYSQL_CUR_TABLE,
                    MILVUS_HOST, MILVUS_PORT,
                    FACE_VECTOR_DIM, FACE_METRIC_TYPE,
                    FACE_INDEX_NLIST, FACE_SEARCH_NPROBE,
                    FACE_INDEX_TYPE, FACE_COLLECTION_NAME)

from server.inference import preprocess, postprocess, run_inference
from server.utils.image import draw_bbox_on_image
import numpy as np
import cv2
import logging

# logger
logger = logging.getLogger('service')


# Connect to MySQL
mysql_conn = pymysql.connect(
    host=MYSQL_HOST,
    port=MYSQL_PORT,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    db=MYSQL_DATABASE,
    cursorclass=DictCursor)

# connect to milvus connec
milvus_collec_conn = get_milvus_collec_conn(
    collection_name=FACE_COLLECTION_NAME,
    milvus_host=MILVUS_HOST,
    milvus_port=MILVUS_PORT,
    vector_dim=FACE_VECTOR_DIM,
    metric_type=FACE_METRIC_TYPE,
    index_type=FACE_INDEX_TYPE,
    index_metric_params={"nlist": FACE_INDEX_NLIST})

# load milvus_collec_conn into memory for faster searches
milvus_collec_conn.load()

# all entity
models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
]


def get_registered_person(
        person_id: int,
        table: str = MYSQL_CUR_TABLE) -> dict:
    """
    Get registered person by person_id.
    Checks redis cache, otherwise query mysql
    """
    # try cached redis data
    # TODO
    # if cache is not found, query mysql
    return select_person_data_from_sql_with_id(
        mysql_conn, table, person_id)


def get_person_avatar(
        person_id: int,
        table: str = MYSQL_CUR_TABLE) -> dict:
    """
    Get the avatar of person by id
    """
    return select_person_avatar_from_sql_with_id(
        mysql_conn, table, person_id
    )


def get_all_registered_person(
        table: str = MYSQL_CUR_TABLE) -> dict:
    """
    Get all registered persons query mysql
    """
    return select_all_person_data_from_sql(mysql_conn, table)


def unregister_person(
        person_id: int,
        table: str = MYSQL_CUR_TABLE) -> dict:
    """
    Deletes a registered person based on the unique person_id.
    Must use expr with the term expression `in` for delete operations
    Operation is atomic, if one delete op fails, all ops fail
    """
    try:
        # unregister from mysql
        # commit is set to False so that the op is atomic with milvus & redis
        mysql_del_resp = delete_person_data_from_sql_with_id(
            mysql_conn, table, person_id, commit=False)
        if mysql_del_resp["status"] == "failed":
            raise pymysql.Error

        # unregister from milvus
        expr = f'person_id in [{person_id}]'
        milvus_collec_conn.delete(expr)
        print("Vector for person with id: %s deleted from milvus db.",
              person_id)

        # clear redis cache
        # todo

        # commit mysql record delete
        mysql_conn.commit()
    except (pymysql.Error, MilvusException) as excep:
        msg = f"person with id {person_id} couldn't be unregistered from database ❌"
        return {"status": "failed",
                "message": msg}
    print("person record with id %s unregistered from database.", person_id)
    return {"status": "success",
            "message": f"person record with id {person_id} unregistered from database"}


def register_person_avatar(
        person_data:dict,
        avatar_path:str,
        table: str = MYSQL_CUR_TABLE):
    """
    Get the user_info and avatar path,insert it into our db
    """
    # uniq person id from user input
    person_id = person_data["id"]
    # check if person already exists in redis/mysql
    if get_registered_person(person_id, table)["status"] == "success":
        # if so, update the avatar
        return update_avatar_url_in_sql(mysql_conn, table, person_id, avatar_path)
    # first time register must set the avatar
    try:
        person_data["avatar_url"] = avatar_path
        # insert record into mysql
        # commit is set to False so that the op is atomic with milvus & redis
        mysql_insert_resp = insert_person_data_into_sql(
            mysql_conn, table, person_data, commit=False)
        if mysql_insert_resp["status"] == "failed":
            raise pymysql.Error

        # commit mysql record insertion(commit after all things done)
        mysql_conn.commit()
        return {"status":"success"}
        
    except (pymysql.Error) as excep:
        msg = f"person with id {person_id} couldn't be registered into database "
        print("error: %s: %s", excep, msg)
        return {"status":"failed"}


def register_person_face(
        person_data: dict,
        face_path: str,
        face_det_threshold: float,
        model_name: str = models[0],
        table: str = MYSQL_CUR_TABLE) -> dict:
    """
    Detects faces in image from the file_path and
    saves the face feature vector & the related person_data dict.
    """
    # uniq person id from user input
    person_id = person_data["id"]
    # check if this person id already exists in redis/mysql
    existed =  get_registered_person(person_id, table)["status"] == "success"
        
    # extract the feature and get the recognition
    pred_dict = run_inference(
        face_path,
        face_feat_model=model_name,
        face_det_thres=face_det_threshold,
        face_count_thres = 1,
        mode="register")

    if pred_dict["status"] == 0 and not pred_dict["face_detections"]:
        return {"status": "failed",
                "message": "No faces were detected in the image"}
    if pred_dict["status"] < 0:
        pred_dict["status"] = "failed"
        return pred_dict
    
    try:

        # insert face_vector into milvus milvus_collec_conn
        face_vector = pred_dict["face_feats"][0]
        data = [[person_id], [face_vector]]

        # if id exsited need to replace the origin face_vector,so delete firstly
        # tip: the vector may not exists for the avatar is more fast
        if existed:
            try:
                expr = f'person_id in [{person_id}]'
                milvus_collec_conn.delete(expr)
            except Exception:
                pass
        # then insert the new vector
        milvus_collec_conn.insert(data)
        logger.info(f"Vector for person with id: %s inserted into milvus db.",
              person_id)
        # After final entity is inserted, it is best to call flush to have no growing segments left in memory
        # flushes collection data from memory to storage
        milvus_collec_conn.flush()

        return {"status":"success"}
    
    except (MilvusException) as excep:
        msg = f"person with id {person_id} couldn't be registered into database "
        print("error: %s: %s", excep, msg)
        return {"status": "failed",
                "message": msg}



def register_person(
        model_name: str,
        face_path: str,
        avatar_path: str,
        face_det_threshold: float,
        person_data: dict,
        table: str = MYSQL_CUR_TABLE) -> dict:
    """
    Detects faces in image from the file_path and
    saves the face feature vector & the related person_data dict.
    person_data dict should be based on the init.sql table schema
    Operation is atomic, if one insert op fails, all ops fail
    """
    # uniq person id from user input
    person_id = person_data["id"]
    # check if face already exists in redis/mysql
    if get_registered_person(person_id, table)["status"] == "success":
        # if so, update the avatar
        return update_avatar_url_in_sql(mysql_conn, table, person_id, avatar_path)

    # extract the feature and get the recognition
    pred_dict = run_inference(
        face_path,
        face_feat_model=model_name,
        face_det_thres=face_det_threshold,
        face_count_thres=1,
        mode="register")

    if pred_dict["status"] == 0 and not pred_dict["face_detections"]:
        return {"status": "failed",
                "message": "No faces were detected in the image"}
    if pred_dict["status"] < 0:
        pred_dict["status"] = "failed"
        return pred_dict

    try:
        # first time register must set the avatar
        person_data["avatar_url"] = avatar_path
        # insert record into mysql
        # commit is set to False so that the op is atomic with milvus & redis
        mysql_insert_resp = insert_person_data_into_sql(
            mysql_conn, table, person_data, commit=False)
        if mysql_insert_resp["status"] == "failed":
            raise pymysql.Error

        # insert face_vector into milvus milvus_collec_conn
        face_vector = pred_dict["face_feats"][0]
        data = [[person_id], [face_vector]]
        print(data)
        milvus_collec_conn.insert(data)
        print(f"Vector for person with id: %s inserted into milvus db.",
              person_id)
        # After final entity is inserted, it is best to call flush to have no growing segments left in memory
        # flushes collection data from memory to storage
        milvus_collec_conn.flush()

        # cache data in redis
        # TODO

        # commit mysql record insertion(commit after all things done)
        mysql_conn.commit()
    except (pymysql.Error, MilvusException) as excep:
        msg = f"person with id {person_id} couldn't be registered into database "
        print("error: %s: %s", excep, msg)
        return {"status": "failed",
                "message": msg}

    print("person record with id %s registered into database.", person_id)
    return {"status": "success",
            "message": f"person record with id {person_id} registered into database"}


def recognize_person(
        file_path: str,
        face_det_threshold: float,
        face_dist_threshold: float = 0.1,
        model_name: str = models[0],
        table: str = MYSQL_CUR_TABLE) -> dict:
    """
    Detects faces in image from the file_path and finds the most similar face vector
    from a set of saved face vectors
    """
    pred_dict = run_inference(
        file_path,
        face_feat_model=model_name,
        face_det_thres=face_det_threshold,
        face_count_thres=1,
        mode="recognize")

    if pred_dict["status"] == 0 and not pred_dict["face_detections"]:
        logger.info(f"No faces were detected in the image {file_path}")
        return {"status": "failed",
                "message": "No faces were detected in the image"}
    if pred_dict["status"] < 0:
        pred_dict["status"] = "failed"
        return pred_dict

    avatars_registered = []
    for ind, face_vector in enumerate(pred_dict["face_feats"]):
        # run a vector search and return the closest face with the L2 metric
        search_params = {"metric_type": "L2", "params": {"nprobe": FACE_SEARCH_NPROBE}}
        results = milvus_collec_conn.search(
            data=[face_vector],
            anns_field="embedding",
            param=search_params,
            limit=3,
            output_fields=["person_id"])
        if not results:
            return {"status": "failed",
                    "message": "no saved face entries found in database"}

        results = sorted(results, key=lambda k: k.distances)

        face_dist = results[0].distances[0]

        person_id = results[0][0].entity.get("person_id")
        logger.info(f"img distance with person: {person_id} is {face_dist}")
        if face_dist > face_dist_threshold:
            logger.info("no similar faces were found in the database")
        else:
            logger.info(f"detected face matches id: {person_id}")
            person_avatar_info = get_person_avatar(person_id, table)

            if person_avatar_info["status"] == "success":
                avatars_registered.append(
                    {
                        "person_id": person_id,
                        "face_pos": pred_dict["face_boxes"][ind],
                        "avatar": person_avatar_info["avatar"]['avatar_url'],
                        "confidence": 1 - (face_dist_threshold - face_dist) / face_dist_threshold
                    }
                )
                print(f"find avatar for {person_id}")
            else:
                print(f"no avatar for {person_id}")

    return {"status": "success", "avatars_pair": avatars_registered}


def replace_face_with_avatar(
        img_path: str,
        save_path: str,
        face_boxes: list,
        avatars: list,
) -> str:
    """
    :param save_path: the save path
    :param img_path: the origin img
    :param face_boxes: ALL detected face locations
    :param avatars: The responded registered avatars
    :return:
    """
    ori_img = cv2.imread(img_path)

    for i, box in enumerate(face_boxes):
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        # resize avatar in order to cover the origin img area
        avatar = cv2.imread(avatars[i])
        avatar_resized = cv2.resize(avatar, (w, h))
        # cover the avatar on original img
        ori_img[y:y + h, x:x + w] = avatar_resized

    cv2.imwrite(save_path, ori_img)


def test_unregister():
    in_face = ["tests/p1.jpg", "tests/p2.jpg", "tests/p3.jpg"]
    for f in range(len(in_face)):
        unregister_person(person_id=f)


def test_register():
    in_face = ["tests/p1.jpg", "tests/p2.jpg", "tests/p3.jpg"]
    in_avatar = ["tests/a1.png", "tests/a2.png", "tests/a3.png"]
    for f in range(len(in_face)):
        register_person(
            model_name=models[0], face_path=in_face[f], avatar_path=in_avatar[f],
            face_det_threshold=0.5, person_data={"id": f}
        )


def test_recognize():
    test_face = ["tests/p1_2.jpg", "tests/ps1.jpg"]
    for tf in test_face:
        print(f"recognizing img {tf}")
        res = recognize_person(
            model_name=models[0],
            file_path=tf,
            face_det_threshold=0.5,
            face_dist_threshold=0.1
        )
        print(res)
        if len(res["avatars_pair"]):
            draw_face_position(tf, tf.replace(".jpg", "_pos.jpg"), res["avatars_pair"])
            boxes = [b["face_pos"] for b in res["avatars_pair"]]
            avatars = [a["avatar"] for a in res["avatars_pair"]]
            replace_face_with_avatar(tf, tf.replace(".jpg", "_covered.jpg"), boxes, avatars)

def draw_face_position(img_path, save_path, pairs):
    orig_cv2_img = cv2.imread(img_path)
    labels, confs, boxes = [], [], []
    for p in pairs:
        labels.append(p["person_id"])
        confs.append(p["confidence"])
        box = p["face_pos"]
        x_min, y_min, y_max, x_max = box["x"], box["y"], box["y"] + box["w"], box["x"] + box["h"]
        boxes.append(np.asarray([x_min, y_min, x_max, y_max]))

    draw_bbox_on_image(orig_cv2_img, boxes, confs, labels)
    cv2.imwrite(save_path, orig_cv2_img)


# if __name__ == '__main__':
#     # test_unregister()
#     # test_register()
#     # test_recognize()
