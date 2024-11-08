"""
File: general.py

Purpose:
For general purposes.

Author: Ernesto Fontes Pupo / Claudia Carballo González
Date: 2024-10-30
Version: 1.0.0
SPDX-License-Identifier: Apache-2.0

"""

import numpy as np
import pandas as pd
import ast
import random

def convert_grid_xy(cell):
    if isinstance(cell, str):
        return ast.literal_eval(cell)
    return cell