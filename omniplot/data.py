
import pandas as pd
import numpy as np
from typing import Union, Dict, List, Optional

class Data_processor:
    
    X: np.ndarray
    
    def __init__(self, df: pd.DataFrame, 
                 category: Union[List, str]=[],
                 values: Union[List, str]=[],):
        pass