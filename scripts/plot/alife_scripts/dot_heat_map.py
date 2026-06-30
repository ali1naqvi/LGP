import re
import subprocess
from collections import defaultdict
from pathlib import Path

re_prog_team = re.compile(r"winner program:\s*(\d+)\s*team:\s*(\d+)")


def count_pairs(log_path: Path) -> dict[tuple[int, int], int]:

    counts = defaultdict(int)

    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            if m := re_prog_team.search(line):
                prog_id = int(m.group(1))
                team_id = int(m.group(2))
                counts[(prog_id, team_id)] += 1

    return counts

def lerp(a: int, b: int, t: float) -> int:
    return round(a + (b - a) * t)


def heat_colour(count: int, max_count: int) -> str:
    if max_count == 0:
        return "#fee0d2"
    light = (0xFE, 0xE0, 0xD2) 
    dark = (0xCB, 0x18, 0x1D)
    t = count / max_count
    r = lerp(light[0], dark[0], t)
    g = lerp(light[1], dark[1], t)
    b = lerp(light[2], dark[2], t)
    return f"#{r:02x}{g:02x}{b:02x}"

re_prog_node = re.compile(
    r'^\s*(P(?P<id>\d+)\s*\[.*?label="\{\s*Program\s+(?P=id)\b.*?)"',
    re.IGNORECASE,
)

re_label_used = re.compile(r'\\nused:\s*\d+') 


def colour_dot(dot_in: Path, dot_out: Path, counts: dict[tuple[int, int], int]) -> None:
    max_count = max(counts.values(), default=0)

    with dot_in.open(encoding="utf-8") as fh_in, dot_out.open("w", encoding="utf-8") as fh_out:
        for line in fh_in:
            m = re_prog_node.search(line)
            if not m:
                fh_out.write(line)
                continue

            prog_id = int(m.group("id"))
            count = sum(v for (p, _), v in counts.items() if p == prog_id)

            colour = heat_colour(count, max_count)
            if "fillcolor=" in line:
                line = re.sub(r'fillcolor="#?[0-9a-fA-F]{6}"', f'fillcolor="{colour}"', line)
            else:
                line = line.rstrip("]\n") + f', fillcolor="{colour}"];\n'

            if re_label_used.search(line):
                line = re_label_used.sub(f'\\nused: {count}', line)
            else:
                line = re.sub(r'(\{[^|}]*Program\s+' + str(prog_id) + r')',
                              rf'\1 \\nused: {count}', line)
            fh_out.write(line)


def with_suffix_stem(p: Path, extra: str) -> Path:
    return p.with_name(p.stem + extra + p.suffix)


def main() -> None:
    root = Path(__file__).resolve().parents[3] / "experiments" / "41_dynamic_rew_obs"
    log_path = root / "graphs" / "log.txt"
    dot_path = root / "graphs" / "gv_mujoco.dot"

    print("Log file :", log_path)
    print("Dot file :", dot_path)

    counts = count_pairs(log_path)
    out_dot = with_suffix_stem(dot_path, "_heatmap")
    colour_dot(dot_path, out_dot, counts)

    out_png = out_dot.with_suffix(".png")
    subprocess.run(
        ["dot", "-Tpng", str(out_dot), "-o", str(out_png)],
        check=True,
    )



if __name__ == "__main__":
    main()
