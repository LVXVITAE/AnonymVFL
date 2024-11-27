import os
import numpy as np
import pandas as pd
class Participant:
    def __init__(self,raw_data_path:str) -> None:
        self.raw_data = pd.read_csv(raw_data_path,header=None)
        pass
