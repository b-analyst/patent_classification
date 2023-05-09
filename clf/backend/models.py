from pydantic import BaseModel
from typing import List, Optional

class Abstract(BaseModel):
    model: str=None
    inp: str=None
    stage_1_thresh: int=None
    stage_2_thresh: float=None

class Validation(BaseModel):
    model: str=None
    stage_1_thresh: int=None
    stage_2_thresh: float=None

# class Stage_1(BaseModel):
#     model: str=None
#     inp: str=None
#     stage_1_thresh: int=None