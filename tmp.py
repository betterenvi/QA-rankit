import sys, os, collections, copy
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

data_fn = 'data/WikiQA-train.tsv'
X = pd.read_csv(data_fn, sep='\t', header=0, dtype=str, skiprows=None, na_values='?', keep_default_na=False)
