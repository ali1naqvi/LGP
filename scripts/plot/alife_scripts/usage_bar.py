import matplotlib.pyplot as plt
import numpy as np

# Sample data for 7 variants
variants = ['Variant A', 'Variant B', 'Variant C', 'Variant D', 'Variant E', 'Variant F', 'Variant G']
total = [100, 150, 90, 120, 200, 175, 110]
used = [40, 130, 60, 115, 140, 90, 85]

x = np.arange(len(variants))

fig, ax = plt.subplots(figsize=(10, 6))

# Plot 'Total' with a wider bar
ax.bar(x, total, width=0.6, label='Total', color='#d3d3d3', edgecolor='black')

# Plot 'Used' with a narrower bar so it appears inside the 'Total' bar
ax.bar(x, used, width=0.3, label='Used', color='#1f77b4', edgecolor='black')

# Add labels, title, and legend
ax.set_ylabel('Quantity')
ax.set_title('Total vs Used by Variant')
ax.set_xticks(x)
ax.set_xticklabels(variants)
ax.legend()

plt.tight_layout()
plt.show()