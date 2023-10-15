#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：x-123
@File    ：app.py
@Author  ：Aidan Lew
@Date    ：2023/10/14 10:00
@Documentation  : http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, Response, status, File, UploadFile
from fastapi.responses import FileResponse
from entity.model import RegFacResult, RegAvaResult,RecResult
from config import (DOWNLOAD_AVATAR_PATH,DOWNLOAD_CACHE_PATH,
                    SERVER_IP,SERVER_PORT,FACE_DET_THRESHOLD,
                    FACE_DIST_THRESHOLD)
from service import register_person_avatar,register_person_face,recognize_person
import os
import logging

app = FastAPI()

logger = logging.getLogger('app')


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/avatars/{filename}")
async def get_avatar(filename: str):
    file_path = f"{DOWNLOAD_AVATAR_PATH}/{filename}"
    return FileResponse(file_path)

@app.post("/register/face/{user_id}", response_model=RegFacResult, status_code=200)
async def register_face(user_id: int, img: UploadFile, response: Response):
    res = {"status": "success", "user_id": user_id}
    try:
        img_name = img.filename
        # just cache the face and delete later
        cache_path = os.path.join(DOWNLOAD_CACHE_PATH, img_name)
        contents = await img.read()
        with open(cache_path, "wb") as f:
            f.write(contents)
        logger.info(f"cache face -> {cache_path}")
    except Exception as e:
        logger.error(f"error cache face: {e}")
        response.status_code = 404
        res["status"] = "failed"
    else:
        # deal with face features into the 
        db_res = register_person_face({"id":user_id},cache_path,FACE_DET_THRESHOLD)
        res["status"] = db_res["status"]
    finally:
        if os.path.isfile(cache_path):
            # delete the cache file
            os.remove(cache_path)
    return res

@app.post("/register/avatar/{user_id}", response_model=RegAvaResult, status_code=200)
async def register_avatar(user_id: int, img: UploadFile, response: Response):
    res = {"status": "success", "user_id": user_id,"img_url":""}
    try:
        img_name = img.filename
        save_path = os.path.join(DOWNLOAD_AVATAR_PATH, img_name)
        contents = await img.read()
        with open(save_path, "wb") as f:
            f.write(contents)
        logger.info(f"save avatar -> {save_path}")
    except Exception as e:
        logger.error(f"error register avatar: {e}")
        response.status_code = 404
        res["status"] = "failed"
    else:
        # save user and avatar into mysql
        db_res = register_person_avatar({"id":user_id},save_path)
        res["status"] = db_res["status"]
        if res["status"] == "success":
            res["img_url"] = SERVER_IP+":"+SERVER_PORT+"/avatars/"+img_name
    return res


@app.post("/recognize/face/{user_id}",response_model=RecResult,status_code=200)
async def recognize_faces(user_id: int, img: UploadFile, response: Response):
    res = {"status": "success", "user_id": user_id,"avatar_pairs":[]}

    try:
        img_name = img.filename
        cache_path = os.path.join(DOWNLOAD_CACHE_PATH, img_name)
        contents = await img.read()
        with open(cache_path, "wb") as f:
            f.write(contents)
        logger.info(f"cache face -> {cache_path}")
    except Exception as e:
        logger.error(f"error cache face: {e}")
        response.status_code = 404
        res["status"] = "failed"
    else:
        # recognize the person
        logger.info(f"recognizing img :{cache_path}")

        rec_res = recognize_person(
                file_path=cache_path,
                face_det_threshold=FACE_DET_THRESHOLD,
                face_dist_threshold=FACE_DIST_THRESHOLD
            )
        res["status"] = rec_res["status"]

        logger.info("recognizing faces fininshed!")
        # if has avatar
        if "avatars_pair" in rec_res.keys():
            for ap in rec_res["avatars_pair"]:
                a_p = {"avatar_url":ap["avatar"],"pos":[]}
                x,y,w,h = ap["face_pos"]['x'],ap["face_pos"]['y'],ap["face_pos"]['w'],ap["face_pos"]['h']
                x1, y1 = x, y
                x2, y2 = x + w, y
                x3, y3 = x, y + h
                x4, y4 = x + w, y + h
                pos = [x1,y1,x2,y2,x3,y3,x4,y4]
                a_p["pos"] = pos
                res["avatar_pairs"].append(a_p) 
    finally:
        if os.path.isfile(cache_path):
            # delete the cache file
            os.remove(cache_path)
    return res