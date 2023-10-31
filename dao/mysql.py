"""
pymysql dao functions
"""
import logging
import pymysql

logger = logging.getLogger('mysql_api')


def update_default_avatar_url_in_sql(mysql_conn, mysql_tb, person_id, avatar_url, commit=True):
    """
    Update avatar_url of a person in a MySQL table
    Note: the transaction must be committed after if commit is False
    """
    query = f"UPDATE {mysql_tb} SET avatar_url = %s WHERE user_id = %s and is_default = 1"
    logger.info(query)
    values = (avatar_url, person_id)
    try:
        with mysql_conn.cursor() as cursor:
            cursor.execute(query, values)
            if commit:
                mysql_conn.commit()
                logger.info("avatar_url updated in MySQL db.")
                return {"status": "success", "message": "avatar_url updated in MySQL db"}
            logger.info("avatar_url update waiting to be committed to MySQL db.")
            return {"status": "success", "message": "avatar_url update waiting to be committed to MySQL db."}
    except pymysql.Error as excep:
        logger.error("%s: MySQL avatar_url update failed", excep)
        return {"status": "failed", "message": "MySQL avatar_url update error"}


def insert_person_data_into_sql(mysql_conn, mysql_tb, person_data: dict, commit: bool = True) -> dict:
    """
    Insert person_data into mysql table with param binding
    Note: the transaction must be commited after if commit is False
    """
    # query fmt: `INSERT INTO mysql_tb (id, col1_name, col2_name) VALUES (%s, %s, %s)`
    query = (f"INSERT INTO {mysql_tb}" +
             f" ({', '.join(person_data.keys())})" +
             f" VALUES ({', '.join(['%s'] * len(person_data))})").replace("'", '')
    logger.info(query)
    values = tuple(person_data.values())
    try:
        with mysql_conn.cursor() as cursor:
            cursor.execute(query, values)
            if commit:
                mysql_conn.commit()
                logger.info("record inserted into mysql db.")
                return {"status": "success",
                        "message": "record inserted into mysql db"}
            logger.info("record insertion waiting to be commit to mysql db.")
            return {"status": "success",
                    "message": "record insertion waiting to be commit to mysql db."}
    except pymysql.Error as excep:
        logger.error("%s: mysql record insert failed", excep)
        return {"status": "failed",
                "message": "mysql record insertion error"}


def select_person_default_avatar_from_sql_with_id(mysql_conn, mysql_tb, person_id: str) -> dict:
    """
    Query mysql db to get person avatar using the uniq person_id
    """
    query = f"SELECT avatar_url FROM {mysql_tb} WHERE user_id = %s and is_default = 1"
    logger.info(query)
    values = person_id
    try:
        with mysql_conn.cursor() as cursor:
            cursor.execute(query, values)
            person_data = cursor.fetchone()
            if person_data is None:
                logger.warning("mysql record with person_id: %s does not exist.", person_id)
                return {"status": "failed",
                        "message": f"mysql record with person_id: {person_id} does not exist"}
            logger.info("Person avatar with person_id: %s retrieved from mysql db.", person_id)
            return {"status": "success",
                    "message": f"record matching person_id: {person_id} retrieved from mysql db",
                    "avatar": person_data
                    }  # 将结果中的avatar字段作为字典返回
    except pymysql.Error as excep:
        logger.error("%s: mysql record retrieval failed", excep)
        return {"status": "failed",
                "message": "mysql record retrieval error"}


def select_person_avatar_from_sql_with_id(mysql_conn, mysql_tb, person_id: str) -> dict:
    """
    Query mysql db to get  person avatar using the uniq person_id
    """
    query = f"SELECT avatar_url FROM {mysql_tb} WHERE user_id = %s"
    logger.info(query)
    values = person_id
    try:
        with mysql_conn.cursor() as cursor:
            cursor.execute(query, values)
            person_data = cursor.fetchall()
            if person_data is None:
                logger.warning("mysql record with person_id: %s does not exist.", person_id)
                return {"status": "failed",
                        "message": f"mysql record with person_id: {person_id} does not exist"}
            logger.info("Person avatar with person_id: %s retrieved from mysql db.", person_id)
            return {"status": "success",
                    "message": f"record matching person_id: {person_id} retrieved from mysql db",
                    "avatar": person_data
                    }  # 将结果中的avatar字段作为字典返回
    except pymysql.Error as excep:
        logger.error("%s: mysql record retrieval failed", excep)
        return {"status": "failed",
                "message": "mysql record retrieval error"}



def select_person_data_from_sql_with_id(mysql_conn, mysql_tb, person_id: str) -> dict:
    """
    Query mysql db to get full person data using the uniq person_id
    """
    query = f"SELECT * FROM {mysql_tb} WHERE user_id = %s"
    logger.info(query)

    values = person_id
    try:
        with mysql_conn.cursor() as cursor:
            cursor.execute(query, values)
            person_data = cursor.fetchall()
            if person_data is None or len(person_data)==0:
                logger.warning("mysql record with person_id: %s does not exist.", person_id)
                return {"status": "success",
                        "message": "mysql record with person_id: {person_id} does not exist"}
            logger.info("Person with person_id: %s retrieved from mysql db.", person_id)
            return {"status": "success",
                    "message": f"record matching person_id: {person_id} retrieved from mysql db",
                    "person_data": person_data}
    except pymysql.Error as excep:
        logger.error("%s: mysql record retrieval failed", excep)
        return {"status": "failed",
                "message": "mysql record retrieval error"}


def select_all_person_data_from_sql(mysql_conn, mysql_tb) -> dict:
    """
    Query mysql db to get all person data
    """
    query = f"SELECT * FROM {mysql_tb}"
    logger.info(query)

    try:
        with mysql_conn.cursor() as cursor:
            cursor.execute(query)
            person_data = cursor.fetchall()
            if person_data is None:
                logger.warning("No mysql person records were found.")
                return {"status": "failed",
                        "message": "No mysql person records were found."}
            logger.info("All persons records retrieved from mysql db.")
            return {"status": "success",
                    "message": "All person records retrieved from mysql db",
                    "person_data": person_data}
    except pymysql.Error as excep:
        logger.error("%s: mysql record retrieval failed", excep)
        return {"status": "failed",
                "message": "mysql record retrieval error"}


def delete_person_data_from_sql_with_id(mysql_conn, mysql_tb, person_id: str, commit: bool = True) -> dict:
    """
    Delete record from mysql db using the uniq person_id
    """
    select_query = f"SELECT * FROM {mysql_tb} WHERE user_id = %s"
    del_query = f"DELETE FROM {mysql_tb} WHERE user_id = %s"
    try:
        with mysql_conn.cursor() as cursor:
            # check if record exists in db or not
            cursor.execute(select_query, (person_id))
            if not cursor.fetchall():
                logger.error("Person with person_id: %s does not exist in mysql db.", person_id)
                return {"status": "failed",
                        "message": f"mysql record with person_id: {person_id} does not exist in db"}

            cursor.execute(del_query, (person_id))
            if commit:
                mysql_conn.commit()
                logger.info("Person with person_id: %s deleted from mysql db.", person_id)
                return {"status": "success",
                        "message": "record deleted from mysql db"}
            logger.info("record deletion waiting to be commited to mysql db.")
            return {"status": "success",
                    "message": "record deletion waiting to be commited to mysql db."}
    except pymysql.Error as excep:
        logger.error("%s: mysql record deletion failed", excep)
        return {"status": "failed",
                "message": "mysql record deletion error"}
