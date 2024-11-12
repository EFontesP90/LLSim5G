"""
File: general.py

Purpose:
For general purposes.

Author: Ernesto Fontes Pupo / Claudia Carballo González
        University of Cagliari
Date: 2024-10-30
Version: 1.0.0
                   GNU LESSER GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

    LLSim5G is a link-level simulator for HetNet 5G use cases.
    Copyright (C) 2024  Ernesto Fontes, Claudia Carballo

"""

import numpy as np
import pandas as pd
import ast
import random

def convert_grid_xy(cell):
    if isinstance(cell, str):
        return ast.literal_eval(cell)
    return cell