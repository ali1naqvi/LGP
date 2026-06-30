import re
import argparse
from pathlib import Path
import os

import numpy as np

import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")

import matplotlib.colors as mcolors

def lighten_color(color, amount=0.5):
    rgb = mcolors.to_rgb(color)
    return tuple(rgb[i] + (1 - rgb[i]) * amount for i in range(3))

STEP_RE   = re.compile(r"Step\s+(\d+)\s+Reward:\s+([+-]?\d+(?:\.\d+)?)")
BREAK_RE  = re.compile(r"Leg break at step\s+(\d+)")

def parse_log(path: Path, steps_per_ep: int):
    rewards, episode, breaks = [], [], []
    current_ep = 0

    with path.open() as fh:
        for line in fh:
            m_step = STEP_RE.search(line)
            m_break = BREAK_RE.search(line)

            if m_step:
                step = int(m_step.group(1))
                reward = float(m_step.group(2))

                if step == 0 and episode:
                    rewards.append(episode)
                    current_ep += 1
                    episode = []

                episode.append(reward)

            elif m_break:
                breaks.append((current_ep, int(m_break.group(1))))

        if episode:
            rewards.append(episode)

    for ep in rewards:
        ep.extend([None] * (steps_per_ep - len(ep)))

    return rewards, breaks

def plot_rewards(rewards, breaks, steps_per_ep, ax, label_prefix,
                 skip_first=True, color=None, smooth_window=1):
    if color is None:
        color = next(ax._get_lines.prop_cycler)["color"]
    num_eps = len(rewards)

    for ep_idx, ep_rewards in enumerate(rewards):
        if skip_first:
            ep_rewards_plot = ep_rewards[1:]
            x_offset = 1
        else:
            ep_rewards_plot = ep_rewards
            x_offset = 0

        if smooth_window and smooth_window > 1:
            arr = np.array(ep_rewards_plot, dtype=float)
            kernel = np.ones(smooth_window) / smooth_window
            ep_rewards_plot = np.convolve(arr, kernel, mode='same').tolist()

        x = [ep_idx * steps_per_ep + s
             for s in range(x_offset, steps_per_ep)]
        ep_label = label_prefix if ep_idx == 0 else None
        ax.plot(x, ep_rewards_plot,
                linewidth=1.4,
                color=color,
                label=ep_label)

    for boundary in range(1, num_eps):
        ax.axvline(boundary * steps_per_ep, color="black",
                   linestyle="--", linewidth=0.8, alpha=0.4)

    leg_break_label = "Leg break" if not hasattr(ax, "_legbreak_plotted") else None

    for ep_idx, step in breaks:
        if skip_first and step == 0:
            continue
        global_step = ep_idx * steps_per_ep + step
        ax.axvline(global_step, color="red", linestyle="--", linewidth=1.2,
                   label=leg_break_label)
        leg_break_label = None  

    ax._legbreak_plotted = True
    #ax.set_xlabel("Global Step")
    #ax.set_ylabel("Reward per Step")
    ax.margins(x=0)

def main():
    parser = argparse.ArgumentParser(description="Plot Ant rewards per step.")
    parser.add_argument("--log_file", type=Path, required=True,
                        help="Path to the text log containing Step/Reward lines.")
    parser.add_argument("--log_file2", type=Path,
                        help="Optional second log file for side‑by‑side comparison.")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes expected in the log (default = 5).")
    parser.add_argument("--steps_per_episode", type=int, default=500,
                        help="Fixed number of steps per episode (default = 500).")
    parser.add_argument("--smooth_window", type=int, default=1,
                        help="Moving average window size for reward smoothing (0 disables smoothing).")
    args = parser.parse_args()

    rewards, breaks = parse_log(args.log_file, args.steps_per_episode)

    rewards = rewards[:args.episodes]
    breaks  = [b for b in breaks if b[0] < args.episodes]

    # optional second run
    fig, ax = plt.subplots(figsize=(6, 4))
    raw_color1 = next(ax._get_lines.prop_cycler)["color"]
    colour_run1 = lighten_color(raw_color1, 0.2)
    plot_rewards(rewards, breaks, args.steps_per_episode,
                 ax=ax, label_prefix="NeuroTPG", color=colour_run1,
                 smooth_window=args.smooth_window)
    if args.log_file2:
        rewards2, breaks2 = parse_log(args.log_file2, args.steps_per_episode)
        rewards2 = rewards2[:args.episodes]
        breaks2  = [b for b in breaks2 if b[0] < args.episodes]
        raw_color2 = next(ax._get_lines.prop_cycler)["color"]
        colour_run2 = lighten_color(raw_color2, 0.2)
        plot_rewards(rewards2, breaks2, args.steps_per_episode,
                    ax=ax, label_prefix="NeuroTPG-Frozen", color=colour_run2,
                    smooth_window=args.smooth_window)

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=13, frameon=True)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    fig.tight_layout()
    out_name = ("episode_steps_graph_compare.pdf"
                if args.log_file2 else "episode_steps_graph.pdf")
    fig.savefig(out_name, format="pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()