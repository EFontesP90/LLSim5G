import numpy as np
import pandas as pd
import ast
import random

def convert_grid_xy(cell):
    if isinstance(cell, str):
        return ast.literal_eval(cell)
    return cell