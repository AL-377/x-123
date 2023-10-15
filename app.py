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
from entity.model import RegResult, RecResult
from config import *
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

@app.post("/register/avatar/{user_id}", response_model=RegResult, status_code=200)
async def register_avatar(user_id: int, img: UploadFile, response: Response):
    res = {"status": "Success", "user_id": user_id}
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
        res["status"] = "Fail"
    else:
        res["img_url"] = SERVER_IP+":"+SERVER_PORT+"/avatars/"+img_name
        # save user and avatar into mysql


    return res

