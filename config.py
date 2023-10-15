#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
configurations and env variables load
"""
import os
from logging.config import dictConfig
from entity.logging import LogConfig

# server info
SERVER_IP = os.getenv("SERVER_IP",default="127.0.0.1")
SERVER_PORT = os.getenv("SERVER_PORT",default="8000")

# save directories
DOWNLOAD_AVATAR_PATH = os.getenv("DOWNLOAD_AVATAR_PATH",default="volumes/avatars")
DOWNLOAD_CACHE_PATH = os.getenv('DOWNLOAD_CACHE_PATH', default="app/.data")
LOG_STORAGE_PATH = os.getenv("LOG_STORAGE_PATH", default="volumes/server_logs")

os.makedirs(DOWNLOAD_AVATAR_PATH,exist_ok=True)
os.makedirs(DOWNLOAD_CACHE_PATH, exist_ok=True)
os.makedirs(LOG_STORAGE_PATH, exist_ok=True)

# logging conf
log_cfg = LogConfig()
# override info & error log paths
log_cfg.handlers["info_rotating_file_handler"]["filename"] = os.path.join(LOG_STORAGE_PATH, "info.log")
log_cfg.handlers["warning_file_handler"]["filename"] = os.path.join(LOG_STORAGE_PATH, "error.log")
log_cfg.handlers["error_file_handler"]["filename"] = os.path.join(LOG_STORAGE_PATH, "error.log")
dictConfig(log_cfg.dict())

# http dao server
API_SERVER_PORT=int(os.getenv("API_SERVER_PORT", default="8080"))

# redis conf
REDIS_HOST = os.getenv("REDIS_HOST", default="0.0.0.0")
REDIS_PORT = int(os.getenv("REDIS_PORT", default="6379"))

# mysql conf
MYSQL_HOST = os.getenv("MYSQL_HOST", default="127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", default="3306"))
MYSQL_USER = os.getenv("MYSQL_USER", default="root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", default="root")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", default="xface")
MYSQL_PERSON_TABLE = os.getenv("MYSQL_PERSON_TABLE", default="person")
# table where ops will be run on
MYSQL_CUR_TABLE = os.getenv("MYSQL_CUR_TABLE", default=MYSQL_PERSON_TABLE)

# milvus conf
MILVUS_HOST = os.getenv("MILVUS_HOST", default="127.0.0.1")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", default="19530"))
FACE_VECTOR_DIM = 2622
FACE_METRIC_TYPE = "L2"
FACE_INDEX_TYPE = "IVF_FLAT" #IVF_PQ  may be best
FACE_COLLECTION_NAME = 'faces'
# num of clusters/buckets for each index specific to IVF_FLAT
FACE_INDEX_NLIST = 4096
# nprobe specific to IVF denotes num of closest buckets/clusters looked into per file
FACE_SEARCH_NPROBE = 2056

# face conf
FACE_DET_THRESHOLD = 0.5
FACE_DIST_THRESHOLD = 0.1

