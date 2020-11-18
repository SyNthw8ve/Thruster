from typing import List

from dataclasses import dataclass

from models.softskill import Softskill
from models.hardskill import Hardskill


@dataclass(frozen=True)
class User:
    name: str
    softSkills: List[Softskill]
    hardSkills: List[Hardskill]
