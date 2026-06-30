import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import sys
import os
import numpy as np

p = pathlib.Path(sys.argv[1])
df = pd.read_csv(p)

plt.rcParams["figure.figsize"] = [7.50, 2.0]

fig, ax = plt.subplots()

cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
c = 0
for column in df:
    ax.plot(df[column], np.ones(20) + c, 'o', label=column, color=cols[0])
    c += 1

# Add labels and title
ax.set_xlabel('MSE')
ax.set_ylabel('')
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(['U of Theil', 'MSE', 'Pearson Correlation'])

plt.ylim(0.5,3.5)
ax.invert_yaxis()
plt.tight_layout()

plt.savefig(p.with_name(p.name.split('.')[0]).with_suffix('.pdf'), format="pdf", bbox_inches="tight")