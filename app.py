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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from entity.model import RegFacResult, RegAvaResult,RecResult
from config import (DOWNLOAD_AVATAR_PATH,DOWNLOAD_CACHE_PATH,
                    SERVER_IP,SERVER_PORT,FACE_DET_THRESHOLD,
                    FACE_DIST_THRESHOLD)
from service import (register_person_avatar,
                    insert_person_avatar,
                    register_person_face,
                    recognize_person,
                    unregister_person,
                    get_person_avatars)
from server.utils.image import remove_bg,remove_padding
import os
import logging

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger('app')

@app.get("/")
async def root():
    return FileResponse("dist/index.html")


@app.get("/avatars/{filename}")
async def get_avatar(filename: str):
    file_path = f"{DOWNLOAD_AVATAR_PATH}/{filename}"
    return FileResponse(file_path)

@app.get("/users/{user_id}/avatars",status_code=200)
async def get_user_avatars(user_id:str, response: Response):
    res = {"status":"success","user_id":user_id}
    try:
        db_res = get_person_avatars(user_id)
        logger.info(f"get avatars of user{user_id} res:{db_res}")
        res["avatars"] = db_res['avatar']
        for ra in res["avatars"]:
            ra["avatar_url"]=ra["avatar_url"][7:]
    except Exception as e:
        logger.error(f"fail to get avatars of user: {user_id}")
        res["status"]="failed"
        response.status_code = 404
    return res

@app.post("/users/{user_id}/avatar",response_model=RegAvaResult, status_code=200)
async def add_user_avatar(user_id:str, img: UploadFile,response: Response):
    res = {"status":"success","user_id":user_id}
    try:
        img_name = img.filename
        if img_name[-4:] != ".jpg":
            img_name += "_"+user_id + ".jpg"
        else:
            img_name = img_name[:-4] + "_"+user_id + ".jpg"
        save_path = os.path.join(DOWNLOAD_AVATAR_PATH, img_name)
        contents = await img.read()
        with open(save_path, "wb") as f:
            f.write(contents)
        logger.info(f"save avatar -> {save_path}")
        # remove the bg and pad
        mid_path = save_path[:-4] + "_mid.jpg"
        dealed_path = save_path[:-4] + "_final.png"
        remove_bg(save_path,mid_path)
        remove_padding(mid_path,dealed_path)
        logger.info(f"save dealed avatar -> {dealed_path}")
    except Exception as e:
        logger.error(f"error add avatar: {e}")
        response.status_code = 404
        res["status"] = "failed"
    else:
        # save user and avatar into mysql
        db_res = insert_person_avatar({"user_id":user_id},save_path)
        res["status"] = db_res["status"]
        if res["status"] == "success":
            res["img_url"] = "/avatars/"+img_name
    return res



@app.post("/register/face/{user_id}", response_model=RegFacResult, status_code=200)
async def register_face(user_id: str, img: UploadFile, response: Response):
    res = {"status": "success", "user_id": user_id}
    try:
        img_name = img.filename
        if ".jpg" not in img_name:
            img_name += ".jpg"
        # just cache the face and delete later
        cache_path = os.path.join(DOWNLOAD_CACHE_PATH, img_name)
        contents = await img.read()
        with open(cache_path, "wb") as f:
            f.write(contents)
        logger.info(f"cache face to register-> {cache_path}")
    except Exception as e:
        logger.error(f"error cache face to register: {e}")
        response.status_code = 404
        res["status"] = "failed"
    else:
        # deal with face features into the 
        db_res = register_person_face({"user_id":user_id},cache_path,FACE_DET_THRESHOLD)
        res["status"] = db_res["status"]
    finally:
        if os.path.isfile(cache_path):
            # delete the cache file
            os.remove(cache_path)
    return res

@app.post("/register/avatar/{user_id}", response_model=RegAvaResult, status_code=200)
async def register_avatar(user_id: str, img: UploadFile, response: Response):
    res = {"status": "success", "user_id": user_id,"img_url":""}
    try:
        img_name = img.filename
        if img_name[-4:] != ".jpg":
            img_name += "_"+user_id + ".jpg"
        else:
            img_name = img_name[:-4] + "_"+user_id + ".jpg"
        save_path = os.path.join(DOWNLOAD_AVATAR_PATH, img_name)
        contents = await img.read()
        with open(save_path, "wb") as f:
            f.write(contents)
        logger.info(f"save avatar -> {save_path}")
        # remove the bg and pad
        mid_path = save_path[:-4] + "_mid.jpg"
        dealed_path = save_path[:-4] + "_final.png"
        remove_bg(save_path,mid_path)
        remove_padding(mid_path,dealed_path)
        logger.info(f"save dealed avatar -> {dealed_path}")
    except Exception as e:
        logger.error(f"error register avatar: {e}")
        response.status_code = 404
        res["status"] = "failed"
    else:
        # save user and avatar into mysql
        db_res = register_person_avatar({"user_id":user_id},save_path)
        logger.info(f"get db_res {db_res}")
        res["status"] = db_res["status"]
        if res["status"] == "success":
            res["img_url"] = "/avatars/"+img_name
    return res

@app.post("/unregister/{user_id}",status_code=200)
async def unregister_person_by_id(user_id:str,response: Response):
    res = unregister_person(user_id)
    if res["status"]!="success":
        response.status_code = 404
    return res

@app.post("/recognize/face/{user_id}",response_model=RecResult,status_code=200)
async def recognize_faces(user_id: str, img: UploadFile, response: Response):
    res = {"status": "success", "user_id": user_id,"avatar_pairs":[]}

    try:
        img_name = img.filename
        if ".jpg" not in img_name:
            img_name += ".jpg"
        cache_path = os.path.join(DOWNLOAD_CACHE_PATH, img_name)
        contents = await img.read()
        with open(cache_path, "wb") as f:
            f.write(contents)
        logger.info(f"cache face to recognize-> {cache_path}")
    except Exception as e:
        logger.error(f"error cache face to recognize: {e}")
        response.status_code = 404
        res["status"] = "failed"
    else:
        # recognize the person
        logger.info(f"recognizing img {cache_path}")

        rec_res = recognize_person(
                file_path=cache_path,
                face_det_threshold=FACE_DET_THRESHOLD,
                face_dist_threshold=FACE_DIST_THRESHOLD
            )
        res["status"] = rec_res["status"]

        logger.info("recognizing faces fininshed!")
        try:
            # if has avatar
            if "avatars_pair" in rec_res.keys():
                for ap in rec_res["avatars_pair"]:
                    filename = ap["avatar"].split("/")[-1]
                    a_p = {"avatar_url":"/avatars/"+filename[:-4]+"_final.png","pos":[]}
                    x,y,w,h,angle = ap["face_pos"]['x'],ap["face_pos"]['y'],ap["face_pos"]['w'],ap["face_pos"]['h'],ap["face_pos"]["angle"]
                    pos = [x,y,w,h,angle]
                    a_p["pos"] = pos
                    res["avatar_pairs"].append(a_p) 
        except Exception as e:
            logger.error(e.with_traceback())
        logger.info(f"recognize return:{res}")
    finally:
        if os.path.isfile(cache_path):
            # delete the cache file
            # face_pos = res["avatar_pairs"][0]["pos"] 
            # in_path = cache_path
            # out_path = os.path.join(DOWNLOAD_CACHE_PATH,"temp.png")
            # draw(face_pos,in_path,out_path)
            os.remove(cache_path)
    return res

def draw(face_pos,in_path,out_path):
    import cv2

    # 读取输入图像
    image = cv2.imread(in_path)

    # 绘制矩形框
    x, y, w, h = face_pos[0],face_pos[1],face_pos[2],face_pos[3]

    cv2.rectangle(image, (x,y), (x+w, y+h), (0, 0, 255), 2)

    # 保存输出图像
    cv2.imwrite(out_path, image)


app.mount("/", StaticFiles(directory="dist"), name="static")
