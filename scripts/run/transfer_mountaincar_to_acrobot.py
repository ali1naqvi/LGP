#!/usr/bin/env python3
"""Transfer existing MountainCarContinuous generation-5000 checkpoints to Acrobot from 5001.

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
from pathlib import Path

from transfer_pendulum_to_acrobot import (
    completed_checkpoint, checkpoint_generation, match_checkpoint_registers, run_stage,
    training_only_config,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIGS = {
    "fixed_rates": "mountaincar_continuous_fixed_rates",
    "fixed_individual": "mountaincar_continuous_fixed_individual",
    "execution_modified_rates": "mountaincar_continuous_execution_modified_rates",
}
GENERATION = 5000


def source_checkpoint(variant: str, seed: int, source_dir: Path) -> Path:
    path = source_dir / SOURCE_CONFIGS[variant] / "checkpoints" / f"cp.{GENERATION}.{seed}.0.rslt"
    if checkpoint_generation(path, seed) != GENERATION:
        raise ValueError(f"Transfer requires generation 5000: {path}")
    return path


def experiment_configs(variant: str) -> tuple[str, str]:
    source = ROOT / "configs" / f"{SOURCE_CONFIGS[variant]}.yaml"
    mountaincar = training_only_config(source.read_text())
    # Existing checkpoint instructions are retained; mutations create scalar ops only.
    mountaincar = re.sub(
        r"^(  )(?:# )?((?:VECTOR_|MATRIX_|SCALAR_VECTOR_|SCALAR_MATRIX_|SCALAR_BROADCAST_OP|OBS_BUFF_SLICE_OP)[A-Z_]*):.*$",
        r"\1\2: 0", mountaincar, flags=re.MULTILINE,
    )
    mountaincar = re.sub(r'^  n_generations:.*$',
                        '  n_generations: 5000',
                        mountaincar, flags=re.MULTILINE)
    mountaincar = mountaincar.replace('# Pendulum training/validation budgets;',
                                      '# MountainCarContinuous training/validation budgets;')
    acrobot = mountaincar.replace('active_tasks: "MountainCarContinuous"',
                                  'active_tasks: "Acrobot"')
    acrobot = re.sub(r'^  n_generations:.*$',
                     '  n_generations: 10000',
                     acrobot, flags=re.MULTILINE)
    acrobot = re.sub(r'^  n_input:.*$',
                     '  n_input: "4"  # Acrobot has four observations',
                     acrobot, flags=re.MULTILINE)
    acrobot = acrobot.replace('Scalar S1 controls force in [-1, 1]',
                                'Scalar S1 controls torque, clamped by the task to [-1, 1]')
    acrobot = acrobot.replace('mountaincar_continuous_', 'acrobot_')
    acrobot = re.sub(r'^  acrobot_max_timesteps:.*$', '  acrobot_max_timesteps: 500',
                    acrobot, flags=re.MULTILINE)
    acrobot = acrobot.replace('# MountainCarContinuous training/validation budgets;',
                                '# Acrobot training/validation budgets;')
    return mountaincar, acrobot


def run_transfer(variant: str, seed: int, args: argparse.Namespace) -> None:
    mountaincar_config, acrobot_config = experiment_configs(variant)
    destination = args.output_dir / f"mountaincar_continuous_to_acrobot_{variant}" / f"seed_{seed}"
    mountaincar_dir, acrobot_dir = destination / "mountaincar_continuous", destination / "acrobot"
    if args.fresh:
        source = mountaincar_dir / "checkpoints" / f"cp.{GENERATION}.{seed}.0.rslt"
        mode = "fresh_mountaincar_continuous"
        description = "fresh MountainCarContinuous 0–5000"
    else:
        source = source_checkpoint(variant, seed, args.source_dir)
        acrobot_config = match_checkpoint_registers(acrobot_config, source)
        mode = "existing_checkpoint"
        description = str(source)
    print(f"{variant} seed {seed}: {description} → Acrobot 5001–{args.generations}; {destination}", flush=True)
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
        "mountaincar_config_sha256": hashlib.sha256(mountaincar_config.encode()).hexdigest(),
        "acrobot_config_sha256": hashlib.sha256(acrobot_config.encode()).hexdigest(),
    }
    manifest_path = destination / "transfer.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f"Existing transfer has different settings: {destination}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.fresh:
        run_stage(args.executable, mountaincar_dir, mountaincar_config, seed, args.seed_aux,
                  GENERATION, args.processes, args.launcher)
    # This is the full population saved after generation-5000 selection.
    if checkpoint_generation(source, seed) != GENERATION:
        raise ValueError(f"Transfer requires generation 5000: {source}")
    checkpoint_dir = acrobot_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    transferred = checkpoint_dir / source.name
    if transferred.exists() and transferred.read_bytes() != source.read_bytes():
        raise ValueError(f"Transferred population differs from MountainCarContinuous: {transferred}")
    if not transferred.exists():
        shutil.copy2(source, transferred)
    run_stage(args.executable, acrobot_dir, acrobot_config, seed, args.seed_aux,
              args.generations, args.processes, args.launcher)
    print(f"Completed {variant} seed {seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", nargs="?", default="execution_modified_rates",
                        choices=[*SOURCE_CONFIGS, "all"],
                        help="Mutation variant (default: execution_modified_rates)")
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int, help="Run one seed")
    seeds.add_argument("--seeds", type=int, nargs="+", help="Selected seeds (default: 1–20)")
    parser.add_argument("--seed-aux", type=int, default=42, help="Simulator seed, matching MountainCarContinuous configs")
    parser.add_argument("--generations", type=int, default=10000, help="Last absolute target generation (10000 gives 5,000 target generations)")
    parser.add_argument("--processes", type=int, default=2, help="MPI processes per run")
    parser.add_argument("--launcher", choices=("mpirun", "srun"), default="mpirun",
                        help="Use srun inside a Slurm allocation")
    parser.add_argument("--jobs", type=int, default=1, help="Simultaneous runs; total MPI processes = jobs × processes")
    parser.add_argument("--executable", type=Path, default=ROOT / "build/release/experiments/TPGExperimentMPI")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "experiments",
                        help="Directory containing the existing MountainCarContinuous experiments")
    parser.add_argument("--fresh", action="store_true",
                        help="Train a new scalar-only MountainCarContinuous population instead of reusing a checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without launching runs")
    args = parser.parse_args()
    if args.generations <= GENERATION:
        parser.error("--generations must exceed 5000")
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
