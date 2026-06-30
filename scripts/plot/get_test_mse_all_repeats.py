import pandas as pd
# import matplotlib.pyplot as plt
import pathlib
import sys
import os
# import numpy as np

directory = sys.argv[1]

print("mse")

for file in os.listdir(sys.argv[1]):
    filename = os.fsdecode(file)
    if filename.endswith(".replay.err"):
        df = pd.read_csv(filename)
        print(df["mse"].mean())
    else:
        continue
