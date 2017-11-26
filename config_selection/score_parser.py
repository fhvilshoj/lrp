import re
import numpy as np
from lrp.configuration import *

class ScoreParser(object):
    def __init__(self, score_file) -> None:
        self.score_file = score_file
        self.config = LRPConfiguration()

    def __str__(self) -> str:
        return self.score_file

    def _infer_config_from_file_name(self):
        pass

