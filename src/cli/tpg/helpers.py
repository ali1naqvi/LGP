import pandas as pd
import subprocess
import os
import click
import re

def get_metrics_from_csv(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Find the maximum best_fitness value
    max_fitness = df["best_fitness"].max()

    # Filter the rows with that best_fitness
    filtered = df[df["best_fitness"] == max_fitness]

    # From those rows, select the one with the lowest generation
    result_row = filtered.loc[filtered["generation"].idxmin()]

    # Extract only the columns of interest
    return result_row[["best_fitness", "generation", "team_id"]].to_dict()   

def get_param_from_yaml_file(parameters_file: str, key: str):
    if not os.path.exists(parameters_file):
        raise click.ClickException(f"Parameters file not found: {parameters_file}")

    # Match e.g. "mj_n_eval_train: 3" or "mj_model_path: \"$TPG/...\""
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*:\s*(.*?)\s*(?:#.*)?$", re.MULTILINE)
    with open(parameters_file, "r", encoding="utf-8") as f:
        text = f.read()

    m = pattern.search(text)
    if not m:
        raise click.ClickException(f"Could not find '{key}:' in {parameters_file}")

    raw = m.group(1).strip()
    # Strip surrounding quotes
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1]
    return raw

def get_int_param_from_yaml_file(parameters_file: str, key: str) -> int:
    raw = get_param_from_yaml_file(parameters_file, key)
    try:
        return int(raw)
    except ValueError as e:
        raise click.ClickException(
            f"Expected integer for '{key}' in {parameters_file}, got: {raw}"
        ) from e

def get_replay_eval_param_names(parameters_file: str, task_to_replay: int = 0):
    active_tasks = get_param_from_yaml_file(parameters_file, "active_tasks")
    tasks = [task.strip() for task in active_tasks.split(",") if task.strip()]
    if not tasks:
        raise click.ClickException(f"No active tasks configured in {parameters_file}")
    if task_to_replay < 0 or task_to_replay >= len(tasks):
        raise click.ClickException(
            f"task_to_replay={task_to_replay} is out of range for active_tasks={active_tasks}"
        )
    task_name = tasks[task_to_replay]
    if task_name.startswith("Mujoco"):
        return "mj_n_eval_train", "mj_n_eval_test"
    if task_name == "FastSimGradient":
        return "gradient_n_eval_train", "gradient_n_eval_test"
    if task_name.startswith("FastSimMaze"):
        return "maze_n_eval_train", "maze_n_eval_test"
    if task_name == "RecursiveForecast":
        return "forecast_n_eval_train", "forecast_n_eval_test"

    raise click.ClickException(
        f"Don't know which replay eval-count parameter to use for task '{task_name}'"
    )

def get_checkpoint_values(seed_tpg, checkpoint_phase=0):
    checkpoint_in_phase = str(int(checkpoint_phase))
    cmd_t = (
        f"grep -iRl end checkpoints/cp.*.{seed_tpg}.{checkpoint_in_phase}.rslt | "
        "cut -d '.' -f 2 | sort -n | tail -n 1"
    )
    checkpoint_in_t = subprocess.check_output(cmd_t, shell=True, text=True).strip()
    return checkpoint_in_phase, checkpoint_in_t

def create_environment_directories(TPG: str, env: str):
    """Create directory structure for environment experiments"""
    
    # Convert environment name to snake_case for directory naming
    env_dir = os.path.join(TPG, "experiments", env)
    
    directories = [
        os.path.join(env_dir, "checkpoints"),
        os.path.join(env_dir, "frames"),
        os.path.join(env_dir, "graphs"),
        os.path.join(env_dir, "logs", "misc"),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create .gitignore in each directory
        gitignore_path = os.path.join(directory, ".gitignore")
        if not os.path.exists(gitignore_path):
            with open(gitignore_path, "w") as f:
                f.write("*\n!.gitignore\n")
                
    return env_dir

def run_single_experiment(executable: str, parameters_file: str, processes: int, seed: int):
    """Run a single TPG experiment with the specified parameters"""
    
    pid = os.getpid()
    stdout_file = f"logs/misc/tpg.{seed}.{pid}.std"
    stderr_file = f"logs/misc/tpg.{seed}.{pid}.err"
    
    cmd = [
        "mpirun", 
        "--oversubscribe",
        "-np", str(processes),
        executable,
        f"parameters_file={parameters_file}",
        f"seed_tpg={seed}",
        f"pid={pid}"
    ]

    click.echo(f"Launching MPI run with command:\n{' '.join(cmd)}")
    click.echo(f"Output will be written to {stdout_file} (stdout) and {stderr_file} (stderr).")

    try:
        with open(stdout_file, 'w') as stdout, open(stderr_file, 'w') as stderr:
            process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
            click.echo(f"Process started in the background with PID: {process.pid}")
    except OSError as e:
        raise click.ClickException(f"Failed to start experiment: {e}")

    click.echo("Evolve (started in background)")

