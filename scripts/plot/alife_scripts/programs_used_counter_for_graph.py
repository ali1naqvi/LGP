#!/usr/bin/env python3
import re
import sys
from collections import defaultdict

prog_team_pat = re.compile(r"winner program:\s*(\d+)\s*team:\s*(\d+)")
step_pat       = re.compile(r"Step\s+(\d+)\b")

def count_per_episode(path: str) -> None:
    episodes          = []                 
    counter           = defaultdict(int)  
    episode_index     = 0                  
    pending_prog_team = None      

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:

            m = prog_team_pat.search(line)
            if m:
                pending_prog_team = (int(m.group(1)), int(m.group(2)))
                continue

            m = step_pat.search(line)
            if m and pending_prog_team:
                step_num = int(m.group(1))

                if step_num == 0 and counter:
                    episodes.append((episode_index, counter))
                    episode_index += 1
                    counter = defaultdict(int)

                counter[pending_prog_team] += 1
                pending_prog_team = None 

    if counter:
        episodes.append((episode_index, counter))

    for idx, counts in episodes:
        print(f"Episode {idx}")
        for (prog, team), n in counts.items():
            print(f"  program {prog} / team {team}: {n}")
        print()  # blank line between episodes


if __name__ == "__main__":
    logfile = "3201.txt"
    count_per_episode(logfile)