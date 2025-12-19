import pandas as pd
import pyarrow.parquet as pq
import sys
import os
import numpy as np
import random as r
import json 

sys.path.insert(0, os.path.join(os.getcwd(), 'chronoepilogi_implementation'))

# Now import the class
from ce_extensions2 import ChronoEpilogi


def OHE_chrono(db, columns,target):
    """T

    """