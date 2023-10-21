import pymysql
from dbutils.pooled_db import PooledDB
from pymysql.cursors import DictCursor


from config import (MYSQL_HOST, MYSQL_PORT,
                    MYSQL_USER, MYSQL_PASSWORD,
                    MYSQL_DATABASE,MYSQL_POOL_SIZE)


# 创建连接池对象
mysql_pool = PooledDB(
    creator=pymysql,
    host=MYSQL_HOST,
    port=MYSQL_PORT,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    database=MYSQL_DATABASE,
    cursorclass=DictCursor,
    maxconnections=MYSQL_POOL_SIZE  # 连接池中允许的最大连接数
)
