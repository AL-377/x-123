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
from entity.model import RegResult, RecResult
from config import *
import os

app = FastAPI()


def record(content: str):
    with open("log.txt", "a") as f:
        f.write(content+"\n")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/register/face/{user_id}", response_model=RegResult, status_code=200)
async def register_face(user_id: int, img: UploadFile, response: Response):
    res = {"status": "Success", "user_id": user_id}
    try:
        img_name = img.filename.encode("utf-8")
        record(img_name.decode("utf-8"))
        save_path = os.path.join(DOWNLOAD_AVATAR_PATH, img_name.decode("utf-8"))
        contents = await img.read()
        with open(save_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        record(str(e))
        response.status_code = 404
        res["status"] = "Fail"
    else:
        res["img_url"] = "".join([SERVER_IP, SERVER_PORT, save_path])
    return res


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename, "content_type": file.content_type}

