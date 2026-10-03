#!/usr/bin/env python3
"""Transfer existing Pendulum generation-1000 checkpoints to Acrobot from 1001.

By default run seeds 1–20 for the selected variant. Use --seed for one run,
--jobs for simultaneous runs, and --dry-run to inspect the experiment plan.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIGS = {
    "fixed_rates": "pendulum_fixed_rates_no_limit",
    "fixed_individual": "pendulum_fixed_individual_all",
    "execution_modified_rates": "pendulum_execution_modified_rates_no_limit",
}
GENERATION = 1000


def completed_checkpoint(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 64))
        return stream.read().rstrip().splitlines()[-1:] == [b"end"]


def checkpoint_generation(path: Path, seed: int) -> int:
    if not completed_checkpoint(path):
        raise ValueError(f"Incomplete checkpoint: {path}")
    header = {}
    with path.open() as stream:
        for line in stream:
            key, _, value = line.strip().partition(":")
            if key in ("t", "seed_tpg"):
                header[key] = int(value)
            if len(header) == 2:
                break
    if header.get("seed_tpg") != seed or "t" not in header:
        raise ValueError(f"Checkpoint seed/generation header is invalid: {path}")
    return header["t"]


def latest_checkpoint(directory: Path, seed: int) -> int | None:
    generations = []
    for path in directory.glob(f"cp.*.{seed}.0.rslt"):
        if completed_checkpoint(path):
            generation = checkpoint_generation(path, seed)
            if path.name != f"cp.{generation}.{seed}.0.rslt":
                raise ValueError(f"Checkpoint filename/header mismatch: {path}")
            generations.append(generation)
    return max(generations, default=None)


def source_checkpoint(variant: str, seed: int, source_dir: Path) -> Path:
    path = source_dir / SOURCE_CONFIGS[variant] / "checkpoints" / f"cp.{GENERATION}.{seed}.0.rslt"
    if checkpoint_generation(path, seed) != GENERATION:
        raise ValueError(f"Transfer requires generation 1000: {path}")
    return path


def match_checkpoint_registers(config: str, checkpoint: Path) -> str:
    # Loading preserves all instructions and constants, including vector ones.
    # Match mutation bounds to the saved layout rather than the edited YAML.
    slots = set()
    with checkpoint.open() as stream:
        for line in stream:
            if line.startswith("MemoryEigen:"):
                fields = line.split(":")
                if fields[2] == "0":
                    slots.add(int(fields[3]))
    if len(slots) != 1:
        raise ValueError(f"Expected a uniform scalar-register layout: {checkpoint}")
    count = slots.pop()
    for key in ("min_initial_mem_slots", "max_initial_mem_slots", "min_memory_slots", "max_memory_slots"):
        config = re.sub(rf"^  {key}:.*$", f"  {key}: {count}  # Match the source checkpoint", config, flags=re.MULTILINE)
    return config


def experiment_configs(variant: str) -> tuple[str, str]:
    source = ROOT / "configs" / "done" / f"{SOURCE_CONFIGS[variant]}.yaml"
    pendulum = source.read_text()
    # Disable new non-scalar instructions; existing checkpoint instructions
    # are preserved by ReadCheckpoint even when their operations are disabled.
    pendulum = re.sub(
        r"^(  )(?:# )?((?:VECTOR_|MATRIX_|SCALAR_VECTOR_|SCALAR_MATRIX_|SCALAR_BROADCAST_OP|OBS_BUFF_SLICE_OP)[A-Z_]*):.*$",
        r"\1\2: 0", pendulum, flags=re.MULTILINE,
    )
    acrobot = pendulum.replace('active_tasks: "Pendulum"', 'active_tasks: "Acrobot"')
    acrobot = re.sub(r'^  n_input:.*$', '  n_input: "4"  # Acrobot has four observations', acrobot, flags=re.MULTILINE)
    acrobot = acrobot.replace('Scalar S1 controls torque, clamped by the task to [-2, 2]',
                              'Scalar S1 controls torque, clamped by the task to [-1, 1]')
    acrobot = acrobot.replace('Standard continuous Pendulum', 'Continuous Acrobot')
    acrobot = acrobot.replace('pendulum_parameters:', 'acrobot_parameters:')
    acrobot = acrobot.replace('pendulum_max_timesteps:', 'acrobot_max_timesteps:')
    acrobot = re.sub(r'^  acrobot_max_timesteps:.*$', '  acrobot_max_timesteps: 500', acrobot, flags=re.MULTILINE)
    acrobot = acrobot.replace('pendulum_n_eval_', 'acrobot_n_eval_')
    return pendulum, acrobot


def run_stage(executable: Path, directory: Path, config: str, seed: int,
              seed_aux: int, last_generation: int, processes: int,
              launcher: str = "mpirun") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoints", "logs", "frames", "plots"):
        (directory / name).mkdir(exist_ok=True)
    config_path = directory / "parameters.yaml"
    if config_path.exists() and config_path.read_text() != config:
        raise ValueError(f"Existing run uses a different config: {directory}")
    config_path.write_text(config)
    checkpoint = latest_checkpoint(directory / "checkpoints", seed)
    if checkpoint is not None and checkpoint >= last_generation:
        return
    launch_command = (["srun", "--ntasks", str(processes), "--kill-on-bad-exit=1"]
                      if launcher == "srun" else
                      ["mpirun", "--oversubscribe", "-np", str(processes)])
    command = launch_command + [str(executable),
               f"parameters_file={config_path}", f"seed_tpg={seed}",
               f"seed_aux={seed_aux}", f"n_generations={last_generation}",
               "checkpoint_in_phase=0", "replay=0", "animate=0"]
    if checkpoint is not None:
        command += ["start_from_checkpoint=1", f"checkpoint_in_t={checkpoint}"]
    else:
        command += ["start_from_checkpoint=0"]
    attempt = str(time.time_ns())
    (directory / "logs" / f"command.{attempt}.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "logs" / f"tpg.{seed}.{attempt}.std").open("w") as stdout, \
         (directory / "logs" / f"tpg.{seed}.{attempt}.err").open("w") as stderr:
        subprocess.run(command, cwd=directory, stdout=stdout, stderr=stderr, check=True)
    final = directory / "checkpoints" / f"cp.{last_generation}.{seed}.0.rslt"
    if checkpoint_generation(final, seed) != last_generation:
        raise ValueError(f"Run did not finish generation {last_generation}: {directory}")


def run_transfer(variant: str, seed: int, args: argparse.Namespace) -> None:
    pendulum_config, acrobot_config = experiment_configs(variant)
    destination = args.output_dir / f"pendulum_to_acrobot_{variant}" / f"seed_{seed}"
    pendulum_dir, acrobot_dir = destination / "pendulum", destination / "acrobot"
    if args.fresh:
        source = pendulum_dir / "checkpoints" / f"cp.{GENERATION}.{seed}.0.rslt"
        mode = "fresh_pendulum"
        description = "fresh Pendulum 0–1000"
    else:
        source = source_checkpoint(variant, seed, args.source_dir)
        acrobot_config = match_checkpoint_registers(acrobot_config, source)
        mode = "existing_checkpoint"
        description = str(source)
    print(f"{variant} seed {seed}: {description} → Acrobot 1001–{args.generations}; {destination}", flush=True)
    if args.dry_run:
        return
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "variant": variant, "seed_tpg": seed, "seed_aux": args.seed_aux,
        "transfer_after_generation": GENERATION,
        "source_config": SOURCE_CONFIGS[variant],
        "source_mode": mode,
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": None if args.fresh else hashlib.sha256(source.read_bytes()).hexdigest(),
        "pendulum_config_sha256": hashlib.sha256(pendulum_config.encode()).hexdigest(),
        "acrobot_config_sha256": hashlib.sha256(acrobot_config.encode()).hexdigest(),
    }
    manifest_path = destination / "transfer.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f"Existing transfer has different settings: {destination}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.fresh:
        run_stage(args.executable, pendulum_dir, pendulum_config, seed, args.seed_aux,
                  GENERATION, args.processes, args.launcher)
    # This is the full population saved after generation-1000 selection.
    if checkpoint_generation(source, seed) != GENERATION:
        raise ValueError(f"Transfer requires generation 1000: {source}")
    checkpoint_dir = acrobot_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    transferred = checkpoint_dir / source.name
    if transferred.exists() and transferred.read_bytes() != source.read_bytes():
        raise ValueError(f"Transferred population differs from Pendulum: {transferred}")
    if not transferred.exists():
        shutil.copy2(source, transferred)
    run_stage(args.executable, acrobot_dir, acrobot_config, seed, args.seed_aux,
              args.generations, args.processes, args.launcher)
    print(f"Completed {variant} seed {seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=[*SOURCE_CONFIGS, "all"])
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int, help="Run one seed")
    seeds.add_argument("--seeds", type=int, nargs="+", help="Selected seeds (default: 1–20)")
    parser.add_argument("--seed-aux", type=int, default=42, help="Simulator seed, matching Pendulum configs")
    parser.add_argument("--generations", type=int, default=5000, help="Last Acrobot generation")
    parser.add_argument("--processes", type=int, default=2, help="MPI processes per run")
    parser.add_argument("--launcher", choices=("mpirun", "srun"), default="mpirun",
                        help="Use srun inside a Slurm allocation")
    parser.add_argument("--jobs", type=int, default=1, help="Simultaneous runs; total MPI processes = jobs × processes")
    parser.add_argument("--executable", type=Path, default=ROOT / "build/release/experiments/TPGExperimentMPI")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "experiments",
                        help="Directory containing the existing Pendulum experiments")
    parser.add_argument("--fresh", action="store_true",
                        help="Train a new scalar-only Pendulum population instead of reusing a checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without launching runs")
    args = parser.parse_args()
    if args.generations <= GENERATION:
        parser.error("--generations must exceed 1000")
    if args.processes < 2 or args.jobs < 1:
        parser.error("--processes must be at least 2 and --jobs at least 1")
    if args.launcher == "srun" and args.jobs != 1:
        parser.error("Use --jobs 1 with srun; use a Slurm array for concurrent seeds")
    selected_seeds = [args.seed] if args.seed is not None else (args.seeds or list(range(1, 21)))
    if len(set(selected_seeds)) != len(selected_seeds) or any(seed < 1 for seed in selected_seeds):
        parser.error("seeds must be distinct positive integers")
    args.output_dir = args.output_dir.resolve()
    args.source_dir = args.source_dir.resolve()
    args.executable = args.executable.resolve()
    if not args.dry_run and (not args.executable.is_file() or shutil.which(args.launcher) is None):
        parser.error(f"Build TPGExperimentMPI from the updated source and ensure {args.launcher} is available")
    variants = list(SOURCE_CONFIGS) if args.variant == "all" else [args.variant]
    jobs = [(variant, seed) for variant in variants for seed in selected_seeds]
    print(f"{len(jobs)} transfers; {args.jobs} simultaneous runs; {args.processes} MPI processes per run", flush=True)
    try:
        # Validate the entire batch before launching any experiment.
        if not args.fresh:
            for variant, seed in jobs:
                source = source_checkpoint(variant, seed, args.source_dir)
                match_checkpoint_registers(experiment_configs(variant)[1], source)
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            results = [executor.submit(run_transfer, variant, seed, args) for variant, seed in jobs]
            for result in results:
                result.result()
    except (ValueError, subprocess.CalledProcessError, OSError) as error:
        parser.exit(1, f"Transfer failed: {error}\n")


if __name__ == "__main__":
    main()
