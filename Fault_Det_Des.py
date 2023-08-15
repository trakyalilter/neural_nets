import os
import pandas as pd
import numpy as np

df = pd.read_csv("/home/ilter-cnc/Downloads/TEP_Faulty_Testing.csv")
print(df.isna().sum())