from typing import NamedTuple, List

from dataclasses import dataclass

from models.softskill import Softskill
from models.hardskill import Hardskill

@dataclass()
class Opening:
    entityId: str
    hardSkills: List[Hardskill]
    softSkills: List[Softskill]
