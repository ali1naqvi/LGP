#!/usr/bin/env python3
import argparse
import re
from collections import OrderedDict
import matplotlib.pyplot as plt
from pathlib import Path

PRESET_PROGRAMS = [233, 590776, 438191, 621378, 675124, 675125]  # only these are counted

def parse_args():
    p = argparse.ArgumentParser(description="Count preset program IDs from a TPG log and plot a bar chart.")
    p.add_argument("logfile", type=Path, help="Path to the txt log file (e.g., /workspaces/tpg/1203.txt)")
    p.add_argument("--out", type=Path, default=Path("program_usage.pdf"),
                   help="Output image file for the bar chart (default: program_usage.png)")
    return p.parse_args()

def main():
    args = parse_args()
    if not args.logfile.exists():
        raise SystemExit(f"File not found: {args.logfile}")

    # Initialize counts (keep given order)
    counts = OrderedDict((pid, 0) for pid in PRESET_PROGRAMS)

    # Regex to capture: winner program: <id>
    prog_re = re.compile(r"winner\s+program:\s*(\d+)", re.IGNORECASE)

    with args.logfile.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = prog_re.search(line)
            if not m:
                continue
            pid = int(m.group(1))
            if pid in counts:
                counts[pid] += 1

    # Print counts to stdout
    print("Program usage counts (only preset IDs):")
    for pid, c in counts.items():
        print(f"{pid}: {c}")

    # Plot
    labels = [str(pid) for pid in counts.keys()]
    values = list(counts.values())
    x = range(len(labels))

    plt.figure(figsize=(8, 4.5))
    plt.bar(x, values)
    plt.xticks(x, labels)
    # plt.xlabel("Program ID")
    # plt.ylabel("Count")
    # plt.title("Distribution of Preset Programs Used")

    # Add counts above bars
    for xi, v in zip(x, values):
        plt.text(xi, v, str(v), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"\nSaved bar chart to: {args.out.resolve()}")

if __name__ == "__main__":
    main()