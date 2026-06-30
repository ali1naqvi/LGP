import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import sys

p = pathlib.Path(sys.argv[1])

plt.rcParams["figure.figsize"] = [7.50, 2.5]
plt.rcParams["figure.autolayout"] = True

df = pd.read_csv(p)

df.plot(x='Time', y=['x0', 'y0'], legend=False)
plt.legend(['Target', 'Prediction'], fontsize=9, ncol=2, frameon=True, loc='upper right', bbox_to_anchor=(1.005, 1.25)) 

# draw prime lines
plt.ylim(0.3,0.85)
plt.hlines(y=0.4, xmin=200, xmax=249, linewidth=2, color='r')
plt.vlines(x=200, ymin=0.4, ymax=0.42, linewidth=2, color='r')
plt.vlines(x=249, ymin=0.4, ymax=0.42, linewidth=2, color='r')
plt.text(225, 0.357, 'prime steps', ha='center', va='center')

plt.title('Title',loc='left')
# plt.title('U of Theil',loc='left')
# plt.title('MSE',loc='left')
# plt.title('Pearson Correlation',loc='left')

#remoxe x axis label on top 2 plots
# plt.xticks(color='w')
# plt.xlabel('Time',color='w')
plt.ylabel('Pitch')

plt.savefig(p.with_name(p.name.split('.')[0]).with_suffix('.pdf'), format="pdf", bbox_inches="tight")