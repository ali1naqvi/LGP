import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

TXT_FILE       = Path("imm_leg.txt")
EXPECTED_STEPS = 500
FILL_ALPHA     = 1.0
LINE_WIDTH     = 0.8
VERT_SPACING   = 1.1
BID_SCALE_FACTOR = 1.0  
NORM_BIDS = True
PLOT_BIDS = True

def softmax_with_epsilon(values, epsilon=1e-3):
    max_val = np.max(values)
    exp_vals = np.exp(values - max_val)
    sum_exps = np.sum(exp_vals)
    adjusted = (exp_vals / sum_exps) * (1.0 - epsilon * len(values)) + epsilon
    return adjusted / np.sum(adjusted)

step_pattern   = re.compile(r"^step\s*$", re.IGNORECASE)
weight_bid_pattern = re.compile(r"Program\s+(\d+):\s+Weight\s*=\s*([\d.eE+-]+)\s+Bid\s*=\s*([\d.eE+-]+)")

program_ids = set()
steps_raw   = []

with TXT_FILE.open() as f:
    current_step = {}
    for line in f:
        if step_pattern.match(line):
            if current_step:
                steps_raw.append(current_step)
            current_step = {}
        else:
            m = weight_bid_pattern.match(line)
            if m:
                pid   = int(m.group(1))
                w_val = float(m.group(2))
                b_val = float(m.group(3))
                current_step[pid] = (w_val, b_val)
                program_ids.add(pid)
    if current_step:
        steps_raw.append(current_step)

steps_raw = steps_raw[:EXPECTED_STEPS]
n_steps    = len(steps_raw)
n_programs = max(program_ids) + 1

weight_data = np.full((n_programs, n_steps), np.nan)
bid_data    = np.full((n_programs, n_steps), np.nan)

for t, record in enumerate(steps_raw):
    for pid, (w, b) in record.items():
        weight_data[pid, t] = w
        bid_data[pid, t]    = b

weight_data = np.nan_to_num(weight_data, nan=0.0)
bid_data    = np.nan_to_num(bid_data, nan=0.0)

if NORM_BIDS:
    for t in range(n_steps):
        bid_data[:, t] = softmax_with_epsilon(bid_data[:, t])


bid_scale = BID_SCALE_FACTOR

x_vals = np.arange(n_steps)

winners = np.argmax(bid_data, axis=0) 
winner_mask = np.equal.outer(np.arange(n_programs), winners) 

fig, ax = plt.subplots(figsize=(13, 9))

for pid in range(n_programs):
    y_off      = pid * VERT_SPACING
    y_vals_w   = weight_data[pid] + y_off

    ax.fill_between(x_vals, y_off, y_vals_w,
                    color="lightgrey", alpha=FILL_ALPHA, linewidth=0)
    ax.plot(x_vals, y_vals_w, color="black", linewidth=LINE_WIDTH)

    ax.fill_between(x_vals, y_off, y_vals_w,
                    where=winner_mask[pid],
                    color="tab:green", alpha=0.9, linewidth=0)

    if PLOT_BIDS:
        y_vals_b = bid_data[pid] * bid_scale + y_off
        ax.plot(x_vals, y_vals_b, linestyle="--", linewidth=0.8, color="tab:blue", alpha=0.9)

ax.set_xlabel("Time step")
ax.set_xlim(0, n_steps - 1)
ax.set_yticks(np.arange(n_programs) * VERT_SPACING)
ax.set_yticklabels([f"Program {i}" for i in range(n_programs)])
ax.set_ylim(-VERT_SPACING, n_programs * VERT_SPACING)
ax.set_title(f"Ridgeline plot of program weights (fills) and scaled bids (dashed blue) over {n_steps} steps")

for spine in ["left", "right", "top"]:
    ax.spines[spine].set_visible(False)
ax.tick_params(axis="y", length=0)

plt.tight_layout()
plt.savefig("weights.pdf", format="pdf", bbox_inches="tight")
plt.show()