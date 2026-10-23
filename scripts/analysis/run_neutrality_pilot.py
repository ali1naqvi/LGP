#!/usr/bin/env python3
"""Build/run a resumable offline assay of saved runtime transfer populations.

Default pilot: both transfers, seeds 1–3, generations 2000–2500 every 100,
20 random elites, 100 offspring, five independent paired episodes.
No training runs or input checkpoints are modified. Standard-library Python only.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
VARIANT = "execution_modified_rates"
BUILD = ROOT / "build/neutrality-pilot"
WORKER = BUILD / "neutrality_worker"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deterministic_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "big") % 2000000000


def generations(value):
    try:
        if ":" in value:
            start, end, step = map(int, value.split(":"))
            if step <= 0 or end < start:
                raise ValueError()
            result = list(range(start, end + 1, step))
        else:
            result = list(map(int, value.split(",")))
        if not result or min(result) < 2000 or len(set(result)) != len(result):
            raise ValueError()
        return sorted(result)
    except ValueError:
        raise argparse.ArgumentTypeError("Use inclusive START:END:STEP or comma-separated generations >= 2000")


def code_hash():
    files = [Path(__file__), ROOT / "scripts/analysis/neutrality_worker.cc"]
    for folder in ("src/engine", "src/environments/classic_control", "src/experiments/evaluators", "src/logging"):
        files.extend(p for p in (ROOT / folder).rglob("*") if p.suffix in (".h", ".cc"))
    h = hashlib.sha256()
    for path in sorted(set(files)):
        h.update(str(path.relative_to(ROOT)).encode()); h.update(path.read_bytes())
    return h.hexdigest()


def build():
    BUILD.mkdir(parents=True, exist_ok=True)
    stamp = BUILD / "build.json"
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    signature = {"code_sha256": code_hash(), "compiler": compiler,
                 "compiler_version": subprocess.check_output(compiler + ["--version"], text=True)}
    if WORKER.exists() and stamp.exists() and json.loads(stamp.read_text()) == signature:
        return
    includes = ["src/engine", "src/experiments", "src/experiments/evaluators", "src/environments",
                "src/environments/classic_control", "src/logging", "modules/libfastsim/src"]
    flags = ["-std=c++23", "-O2", "-DHPCC", "-Wno-deprecated-declarations", "-Wno-format",
             "-ffunction-sections", "-fdata-sections"]
    if sys.platform == "darwin":
        prefix = subprocess.check_output(["brew", "--prefix"], text=True).strip()
        includes += [f"{prefix}/include", f"{prefix}/include/eigen3"]
        links = [f"-L{prefix}/lib", "-Wl,-dead_strip"]
        sdk = Path(subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip())
        gl = BUILD / "include/GL"; gl.mkdir(parents=True, exist_ok=True)
        for header in ("gl.h", "glu.h", "glext.h", "glut.h"):
            target = sdk / f"System/Library/Frameworks/{'GLUT' if header == 'glut.h' else 'OpenGL'}.framework/Headers" / header
            link = gl / header
            if not link.is_symlink():
                link.symlink_to(target)
        includes.append(str(gl.parent))
    else:
        includes += ["/usr/include/eigen3"]
        flags += shlex.split(subprocess.check_output(["pkg-config", "--cflags", "ompi-cxx", "eigen3", "yaml-cpp"], text=True))
        links = ["-Wl,--gc-sections"]
    sources = ["scripts/analysis/neutrality_worker.cc", "src/engine/RegisterMachine.cc", "src/engine/TPG.cc",
               "src/engine/instruction.cc", "src/engine/misc.cc", "src/engine/team.cc", "src/engine/hebbian.cc"]
    command = compiler + flags + [f"-I{x}" for x in includes] + sources + links
    command += ["-lboost_iostreams", "-lyaml-cpp", "-o", str(WORKER)]
    print("Building native worker…", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    stamp.write_text(json.dumps(signature, indent=2) + "\n")


def plan(args):
    jobs = []
    for task in args.tasks:
        for seed in args.seeds:
            base = ROOT / f"experiments/mountaincar_continuous_to_{task}_{VARIANT}/seed_{seed}"
            for generation in args.generations:
                stage = "mountaincar_continuous" if generation == 2000 else task
                config = base / stage / "parameters.yaml"
                checkpoint = base / stage / f"checkpoints/cp.{generation}.{seed}.0.rslt"
                target_config = base / task / "parameters.yaml" if generation == 2000 else None
                paths = [config, checkpoint] + ([target_config] if target_config else [])
                for path in paths:
                    if not path.is_file():
                        raise ValueError(f"Missing required input: {path}; choose available seeds/checkpoints")
                if generation == 2000:
                    copied = base / task / f"checkpoints/cp.2000.{seed}.0.rslt"
                    if not copied.exists() or digest(checkpoint) != digest(copied):
                        raise ValueError(f"Transfer checkpoint identity check failed: {base}")
                jobs.append(dict(task=task, seed=seed, generation=generation,
                    config=str(config), checkpoint=str(checkpoint),
                    target_config=str(target_config) if target_config else None,
                    input_sha256={str(p): digest(p) for p in paths},
                    parents=args.parents, offspring=args.offspring, episodes=args.episodes,
                    sampling_seed=deterministic_seed(args.master_seed, "sample", task, seed, generation),
                    evaluation_seed=deterministic_seed(args.master_seed, "evaluation", task, seed),
                    mutation_seed=deterministic_seed(args.master_seed, "mutation", task, seed, generation),
                    reset_policy="fresh inherited-constant initialization for every independent episode",
                    mutation_state="checkpoint self_modifying_state; native CloneProgram/ProgramMutator_Instructions"))
    return jobs


def summarize(directory, expected_episodes):
    with (directory / "individuals.csv").open() as f:
        individuals = {(int(r["parent_index"]), int(r["offspring"])): r for r in csv.DictReader(f)}
    hashes = {}
    with (directory / "genomes.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            hashes[(r["parent_index"], r["offspring"])] = hashlib.sha256(r["genotype"].encode()).hexdigest()
    groups = {}
    with (directory / "episodes.csv").open() as f:
        for row in csv.DictReader(f):
            key = (int(row["parent_index"]), int(row["offspring"]), row["environment"])
            groups.setdefault(key, []).append(row)
    results = []
    for (parent, child, environment), rows in groups.items():
        if len(rows) != expected_episodes or len({r["episode"] for r in rows}) != expected_episodes:
            raise ValueError(f"Incomplete or duplicate episodes: {directory}")
        if child < 0:
            continue
        record = dict(individuals[parent, child], environment=environment,
                      genotype_sha256=hashes[parent, child], parent_genotype_sha256=hashes[parent, -1], n_episodes=len(rows))
        for role in ("total", "action_effective", "mutation_only", "introns"):
            record[f"parent_{role}"] = individuals[parent, -1][role]
            record[f"delta_{role}"] = int(record[role]) - int(record[f"parent_{role}"])
        for rate in ("swap", "delete", "add", "mutate"):
            parent_rows = groups[parent, -1, environment]
            record[f"mean_rate_delta_{rate}"] = (sum(float(r[f"mean_{rate}"]) for r in rows)
                 - sum(float(r[f"mean_{rate}"]) for r in parent_rows)) / len(rows)
        for field in ("delta_return", "return", "parent_return", "probe_action_mae",
                      "probe_swap_mae", "probe_delete_mae", "probe_add_mae", "probe_mutate_mae"):
            values = [float(r[field]) for r in rows]
            record[f"mean_{field}"] = sum(values) / len(values)
        deltas = [float(r["delta_return"]) for r in rows]
        mean = record["mean_delta_return"]
        record["sd_delta_return"] = (sum((v - mean)**2 for v in deltas) / (len(deltas) - 1))**0.5 if len(deltas) > 1 else ""
        results.append(record)
    with (directory / "offspring_summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    return len(groups), len(results)


def run_job(job, output, signature):
    directory = output / job["task"] / f"seed_{job['seed']}" / f"generation_{job['generation']}"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "worker.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"Another worker is using {directory}")
        run_locked_job(job, output, directory, signature)


def run_locked_job(job, output, directory, signature):
    manifest = dict(job, code_sha256=signature, worker_sha256=digest(WORKER))
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f"Inputs/settings/code changed: choose a new --output directory ({directory})")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    complete = directory / "complete.json"
    if complete.exists():
        stored = json.loads(complete.read_text())
        if all((directory / name).is_file() and digest(directory / name) == sha for name, sha in stored["output_sha256"].items()):
            print(f"Already complete: {directory.relative_to(output)}", flush=True); return
        raise ValueError(f"Completed outputs changed: {directory}")
    command = [str(WORKER), job["config"], job["checkpoint"], job["target_config"] or "-"]
    command += [str(job[k]) for k in ("seed", "generation", "parents", "offspring", "episodes", "sampling_seed", "evaluation_seed", "mutation_seed")]
    print(f"Running {directory.relative_to(output)}", flush=True)
    start = time.monotonic()
    with (directory / "worker.log").open("w") as log:
        subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True)
    groups, children = summarize(directory, job["episodes"])
    environments = 2 if job["target_config"] else 1
    expected = job["parents"] * (job["offspring"] + 1) * environments
    if groups != expected:
        raise ValueError(f"Incomplete worker output: {directory}")
    for name, expected_rows in (("individuals.csv", job["parents"] * (job["offspring"] + 1)),
                                ("episodes.csv", expected * job["episodes"])):
        with (directory / name).open() as f:
            if sum(1 for _ in csv.DictReader(f)) != expected_rows:
                raise ValueError(f"Unexpected record count: {directory / name}")
    files = ["individuals.csv", "episodes.csv", "genomes.jsonl", "offspring_summary.csv"]
    complete.write_text(json.dumps(dict(seconds=time.monotonic() - start, offspring_summaries=children,
        output_sha256={name: digest(directory / name) for name in files}), indent=2) + "\n")
    print(f"Completed {directory.relative_to(output)} ({time.monotonic()-start:.1f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "build", "run", "smoke"))
    parser.add_argument("--tasks", nargs="+", choices=("pendulum", "acrobot"), default=["pendulum", "acrobot"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--generations", type=generations, default=generations("2000:2500:100"))
    parser.add_argument("--parents", type=int, default=20)
    parser.add_argument("--offspring", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--master-seed", type=int, default=20261019)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent checkpoint jobs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.parents, args.offspring, args.episodes, args.workers) < 1:
        parser.error("Counts must be positive")
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate tasks/seeds are not allowed")
    if args.command == "build":
        build(); print(WORKER); return
    if args.command == "smoke":
        args.seeds = [1]; args.generations = [2000, 2100]; args.parents = 1; args.offspring = 2; args.episodes = 2
    output = (args.output or ROOT / "output" / ("neutrality_pilot_smoke" if args.command == "smoke" else "neutrality_pilot")).resolve()
    jobs = plan(args)
    episodes = sum(j["parents"] * (j["offspring"] + 1) * j["episodes"] * (2 if j["target_config"] else 1) for j in jobs)
    print(f"{len(jobs)} checkpoint jobs; {episodes:,} closed-loop episodes, plus shared-history probes. Output: {output}")
    if args.command == "plan":
        for job in jobs:
            print(f"  {job['task']} seed {job['seed']} generation {job['generation']}")
        return
    build()
    output.mkdir(parents=True, exist_ok=True)
    signature = code_hash()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda job: run_job(job, output, signature), jobs))
    print(f"Finished. Per-checkpoint offspring_summary.csv files are under {output}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, subprocess.CalledProcessError) as error:
        print(f"Pilot failed: {error}. Inspect the checkpoint's worker.log for native errors.", file=sys.stderr)
        sys.exit(1)
