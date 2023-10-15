"""
data entity for fastapi+uvicorn server
"""
from datetime import date
from enum import Enum
from pydantic import BaseModel

class RegFacResult(BaseModel):
    """
    The response for register the person face
    """
    status: str
    user_id: int
 

class RegAvaResult(BaseModel):
    """
    The response for register the avatar
    img_url: str the url of the avatar img
    """
    status: str
    user_id: int
    img_url: str = ""


class AvatarPair(BaseModel):
    """
    avatar_url: avatar url
    pos: the position at lt_x,lt_y,ld_x,ld_y,rt_x,rt_y,rd_x,rd_y
    """
    avatar_url:str
    pos:list[int]


class RecResult(BaseModel):
    """
    The response for recognize data model
    img_url: str the url of the covered img
    """
    status: str
    user_id: int
    avatar_pairs: list[AvatarPair]


class PersonModel(BaseModel):
    """
    Person data model. Based on the person table schema
    id: int = must be a unique id in the database, required
    name: str = name of person, required
    birthdate: str = date with format YYYY-MM-DD, required
    country: str = country, required
    city: str = city, optional
    title: str = person's title, optional
    org: str = person's org, optional
    """
    ID: int
    name: str
    birthdate: date
    country: str
    city: str = ""
    title: str = ""
    org: str = ""


class InputModel(BaseModel):
    """
    API input model format
    """
    model_name: str
    file_path: str
    face_det_threshold: float = 0.3
    face_dist_threshold: float = 10
    person_data: PersonModel = None


class ModelType(str, Enum):
    """
    Face feature extraction model type
    """
    FAST = "face-reidentification-retail-0095"
    SLOW = "facenet_trtserver"
