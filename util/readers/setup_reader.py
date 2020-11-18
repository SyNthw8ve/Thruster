import os
import logging

from models.user_transformer import UserInstance
from models.opening_transformer import OpeningInstance

from typing import List


class DataInitializer():

    def __init__(self):

        pass

    @staticmethod
    def read_users(users_instances_path: str) -> List[UserInstance]:

        users_instances = []

        if os.path.exists(users_instances_path):

            logging.info("Users instances file found. Loading...")
            users_instances = UserInstance.load_instances(users_instances_path)

        else:
            logging.info("Users instances file not found.")

        return users_instances

    @staticmethod
    def read_openings(openings_instances_path: str) -> List[OpeningInstance]:

        openings_instances = []

        if os.path.exists(openings_instances_path):

            logging.info("Openings instances file found. Loading...")
            openings_instances = OpeningInstance.load_instances(
                openings_instances_path)

        else:

            logging.info(
                "Openings instances file not found.")

        return openings_instances
