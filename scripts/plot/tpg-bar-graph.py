import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update({
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    # 'axes.labelsize': 16,
    # 'axes.titlesize': 18,
    # 'legend.fontsize': 14
})

# Data
labels = [
    'Ant-V2: NeuroTPG Stateful',
    'Ant-V2: NeuroTPG Stateless',
    'Ant-V2 Immediate Breaking: NeuroTPG Stateful',
    'Ant-V2 Immediate Breaking: NeuroTPG Stateless'
]
values = [
    1607 / 2502,
    2500 / 2500,
    1779 / 2500,
    738  / 2504
]
y_pos = range(len(labels))

# Assign one distinct color per bar from the default cycle
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.grid(which="both", color="lightgray", 
            linewidth=0.5, alpha=0.6)
bars = ax.barh(y_pos, values, alpha=0.65, height=0.5)
# Ensure each bar uses its designated color
for bar, color in zip(bars, colors):
    bar.set_color(color)

# Add probability labels inside bars
for bar in bars:
    width = bar.get_width()
    ax.text(
        width / 2,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.3f}",
        va="center",
        ha="center",
        color="white",
        fontsize=14,
        fontweight='bold'
    )

# Labels & ticks

#ax.set_xlabel('Proportion')
#ax.set_title('Comparison of Stateful vs Stateless Neuro Variants')

# Legend: use the bars themselves so colors match
ax.legend(bars, labels, loc='upper right', fontsize=13, frameon=False)

ax.set_yticks([]) 
# Optional grid on x-axis
ax.xaxis.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
plt.savefig("bar_graph.pdf", format="pdf", bbox_inches="tight")