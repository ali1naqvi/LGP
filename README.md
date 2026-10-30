# NEURO Tangled Program Graphs
This code reproduces results from the paper: 

Stephen Kelly, Tatiana Voegerl, Wolfgang Banzhaf, and Cedric Gondro. Evolving Hierarchical Memory-Prediction Machines in Multi-Task Reinforcement Learning. Genetic Programming and Evolvable Machines, 2021. [pdf](https://rdcu.be/czd3s)

This branch is curated for Self-Modifying Mutation Rates. For the additional parameters introduced, please look at the parameters in config files and adjust based on your own task. All tasks being updated on main should work!

## Quick Start

This code is designed to be used in Linux. If you use Windows, you can use Windows Subsystem for Linux (WSL). You can work with WSL in Visual Studio Code by following [this tutorial](https://code.visualstudio.com/docs/remote/wsl-tutorial). Run this to automatically install all dependencies and compile:

```bash
bash ./setup.sh
```

This performs the setup and compilation of the steps below. If you want to manually install, follow the instructions below.

For MacOS or Windows users, you can follow this [guide](https://gitlab.cas.mcmaster.ca/kellys32/tpg/-/wikis/Dev-Container-Setup-Guide) to setup Dev Containers which spins up a Linux based environment right within VS Code.

### 1. Install required software

From the tpg directory run:

```
sudo xargs --arg-file requirements.txt apt install
```
From [MuJoco](https://mujoco.org/), choose what version you would like and update the commands specific to your environment:

```
wget https://github.com/deepmind/mujoco/releases/download/3.2.2/mujoco-3.2.2-linux-aarch64.tar.gz
tar -xzf mujoco-3.2.2-linux-x86_64.tar.gz
```

Since there is also the addition of a maze environment, use this at the start:
```
git submodule update --init --recursive
```
and this after changing up the environment
```
git submodule update --remote --merge
```

### 2. Set environment variables for DRA

In order to easily access tpg scripts, we add appropriate folders to the $PATH environment variable.
To do so, add the following to _~/.profile_

```
export TPG=<YOUR_PATH_HERE>/tpg
export PATH=$PATH:$TPG/scripts/plot
export PATH=$PATH:$TPG/scripts/run
export MUJOCO=<YOUR_PATH_TO_MUJOCO>/mujoco-3.2.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCO/lib/
```

(For reference, I run):
```
export TPG=/root/tpg
export PATH=$PATH:$TPG/scripts/plot
export PATH=$PATH:$TPG/scripts/run
export MUJOCO=/root/mujoco-3.2.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCO/lib/
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
```

Then run:

```
source ~/.profile
```

### 3. Compile

From the tpg directory run:

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

To run in debug mode:

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

To run the build with compiler optimization flags:

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DENABLE_HIGH_OPTIMIZATION=ON
cmake --build build
```

Additionally, you can use the `Makefile` to build TPG as well.

To build in debug mode:

```bash
make debug
```

To build in release mode:

```bash
make release
```

To build with optimized flags:

```bash
make optimized
```

To clean the Build:

```bash
make clean
```

### 4. Run an experiment

Refer to the [wiki](https://gitlab.cas.mcmaster.ca/kellys32/tpg/-/wikis/Running-Experiments-with-the-TPG-CLI) for more information on how to run experiments with the CLI

Use this to run an experiment (where the name here is your yaml file): 
```
tpg evolve inverted_pendulum
```

for multiple tests: 
```
for seed in `seq 1 3`; do tpg-run-mpi.sh -n 5 -s $seed; done
```

For the DRA, I use:
```
for seed in `seq 2 21`; do sbatch $TPG/scripts/run/tpg-run-slurm_short.sh -s $seed -p $TPG/configs/gradient_test_perception_4_peaks.yaml; done
```


### Validation / Test cadence (every N generations)

TPG supports three evaluation phases: **train**, **validation**, and **test**.

- **How many episodes per phase** is controlled by task parameters (e.g. for MuJoCo: `mj_n_eval_train`, `mj_n_eval_validation`, `mj_n_eval_test`).
- **How often validation/test runs during training** is controlled by GA parameters:
  - `validation_mod`: run validation every N generations (set to `0` to disable)
  - `test_mod`: run the (validation + test) procedure every N generations (set to `0` to disable)

If you want **validation every 50 generations** without running test, set:

```
validation_mod: 50
test_mod: 0
mj_n_eval_validation: <nonzero>
```

### Self-modifying register episode reset

Set `reset_self_modifying_per_episode: 1` in the same YAML section as
`self_modifying` to reset S2–S5 (mutation rates) and S6 (decoy) to their
inherited constants at the start of every episode. The default, `0`, preserves
their working values across episodes. This applies only to self-modifying
programs, including registers marked stateful; within-episode behavior is
unchanged.

This toggle is independent of `reset_per_episode` (Hebbian weights) and
`reset_self_modifying_before_mutation` (which rates offspring mutation uses).
With episode resets enabled and reset-before-mutation disabled, mutation uses
the final episode's output rates. You can also pass
`reset_self_modifying_per_episode=1` on the command line after `parameters_file`.

### 5. Plot results

New selection CSVs include `self_modification_only_instruction_count`: the
number of effective instructions retained exclusively for mutation-rate outputs
(S2–S5), including their dependencies. Instructions also needed for bid/action
outputs remain in the action count, and the S6 decoy is excluded. The metric is
zero when self-modification is disabled. Existing effective counts are unchanged.
Subtract it from `effective_program_instruction_count` to obtain the effective
instruction count for bid/action outputs. Both counts use the same policy traversal.
Older CSVs do not contain this metric.

Core regression checks can be run with:

```sh
cmake -S . -B build/metrics -DTPG_BUILD_EXPERIMENTS=OFF -DTPG_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/metrics --target self_modification_metrics_test
ctest --test-dir build/metrics --output-on-failure
```

Generate classic_control_p0.pdf with various statistics:

```
tpg-plot-evolve.py all-selection all
```

The first page will be a training curve looking something like the plot below. A fitness of ~1000 indicates the agent balances the pole for 1000 timesteps, thus solving the task.

<img src="./images/MuJoco_Inverted_Pendulum_Fitness.png" height="300" />

### 6. Visualize the best policy's behaviour for chemotaxis

Turn the option for saving a video on through:
```
gradient_save_video: 1
```

```
tpg replay 1_gradient_baldwin
```

### 7. Cleanup

Delete all checkpoints and output files:

```
tpg-cleanup.sh
```

### Self-modifying register layout (four rates)

Redundancy mutation has been removed. S2–S5 hold swap, delete, add, and point
mutation rates; S6 is the isolated decoy. It cannot be read or written by policy
instructions and is excluded from effective instruction/register counts.
Self-modifying configurations have one fewer memory slot to preserve their
ordinary scalar-register capacity. New checkpoints carry an `SL4` layout marker;
legacy self-modifying checkpoints cannot be resumed or replayed with this build.
Use fresh runs for comparisons: existing logs still reflect the previous engine.
Selection and replay CSVs no longer emit redundancy-rate columns.

### Pendulum to Acrobot transfer

Rebuild `TPGExperimentMPI` from the updated source before running transfers:

```bash
cmake --build build --config Release
python3 scripts/run/transfer_pendulum_to_acrobot.py execution_modified_rates
```

The runner defaults to execution-modified mutation rates, evolutionary seeds
1–20, and simulator seed 42. This variant uses `self_modifying: 1` and
`reset_self_modifying_before_mutation: 0`, so parent execution outputs control
offspring mutation rates. Each run
loads the existing full-population training checkpoint at generation 1000, then
evaluates the full surviving population on Acrobot at generation 1001 and continues
through generation 5000. For `fixed_rates`, the source is
`experiments/pendulum_fixed_rates_no_limit/checkpoints/cp.1000.<seed>.0.rslt`.
The fixed-individual source is `pendulum_fixed_individual_all`, and the
execution-modified source is `pendulum_execution_modified_rates_no_limit`. The first
Acrobot evaluation establishes fitness before further reproduction. The source
mutation settings, constants, state flags, and inherited mutation rates are
preserved; register bounds match the saved population rather than the edited YAML.
Acrobot receives four observations and uses its own torque bounds.
Pendulum uses 200 steps and Acrobot uses 500 steps per episode. Both use
20 training episodes and 5 validation episodes. Validation
runs every 100 generations with separate episode seeds. The archived fixed-rate
Pendulum generation-1000 checkpoints contain vector operations; the fixed-individual
and execution-modified checkpoints for seeds 1–20 contain only scalar instructions.
Inherited instructions are retained exactly, including vector instructions. The
Acrobot config disables creation of new vector/matrix operations. Use `--fresh`
to train new scalar-only Pendulum source populations through generation 1000 instead.

Use `fixed_individual` or `fixed_rates` for the other variants, or
`all` for 20 runs of each variant (60 transfers). `--seed 2` selects one run,
`--seeds 1 2 3` selects a subset, and `--jobs 4 --processes 2` launches four runs
at once using eight MPI processes. `--generations 2000` ends Acrobot at generation
2000; this is an absolute generation number. `--dry-run` prints the plan.
`--source-dir` selects another directory containing the Pendulum experiments.
Every requested source checkpoint is checked before any Acrobot run is launched.

Outputs are saved under
`experiments/pendulum_to_acrobot_<variant>/seed_<seed>/acrobot`; a `pendulum`
subdirectory is used only with `--fresh`. Source checkpoints are copied and their
hashes recorded; the original Pendulum experiment files are not modified.
Rerunning the same command resumes completed checkpoints and skips completed
stages. Each run records its configs and transfer settings, and every launch gets
separate logs. A changed config or simulator seed is rejected for an existing run;
choose a new `--output-dir` for a different experiment.

Transfer verification:

```bash
python3 -m unittest discover -s tests -p test_transfer_runner.py
python3 scripts/run/test_checkpoint_transfer.py
```

The second test compiles the engine and checks the generation boundary and
population preservation against generation-1000 checkpoints for the three
Pendulum variants. It also checks Acrobot's observation count, configurable
episode budgets, and simulation steps.

For an Alliance Slurm cluster, use the module environment you normally use to
build and run TPG (including Python 3.10 or newer). Update the server copy of the
repository and make sure its Pendulum experiment directories contain the
generation-1000 checkpoints for seeds 1–20. Then submit from the repository root:

```bash
export TPG="$PWD"
cmake --build build --config Release
sbatch scripts/run/transfer-pendulum-acrobot.slurm execution_modified_rates
```

The Slurm file uses the existing `def-skelly` account, 21 MPI tasks per seed,
6 GB per CPU, a 23-hour limit, and an array of seeds 1–20 with four running at once.
It launches the Python driver once per seed; the driver starts the executable with
`srun`. It does not launch a Python driver for every MPI rank.

Submit the other variants separately for 20 runs each:

```bash
sbatch scripts/run/transfer-pendulum-acrobot.slurm fixed_individual
sbatch scripts/run/transfer-pendulum-acrobot.slurm fixed_rates
```

Slurm stdout/stderr appear in the submission directory as
`transfer-<array-job-id>_<seed>.out` and `.err`. Detailed experiment logs remain
in each seed's `acrobot/logs` directory. Use `squeue -u "$USER"` to monitor jobs.
Resubmit the same command after a timeout to resume from the latest complete
checkpoint; finished seeds are skipped. Avoid submitting the same variant and
seed twice concurrently.

Resource and seed overrides can be passed to `sbatch` before the script path:

```bash
# Retry only seeds 3 and 7, or reduce the simultaneous seed count.
sbatch --array=3,7 scripts/run/transfer-pendulum-acrobot.slurm fixed_rates
sbatch --array=1-20%2 scripts/run/transfer-pendulum-acrobot.slurm fixed_rates
```

### MountainCarContinuous to Pendulum transfer

Reuse the full-population MountainCarContinuous training checkpoints at generation
2000 (2,001 generations numbered 0–2000). Pendulum starts at generation 2001 and
finishes at generation 3000, giving exactly 1,000 Pendulum generations. The first
Pendulum generation evaluates the restored population before reproduction.

```bash
cmake --build build --config Release
python3 scripts/run/transfer_mountaincar_to_pendulum.py execution_modified_rates
# Check all 60 source checkpoints and print the plan without running training:
python3 scripts/run/transfer_mountaincar_to_pendulum.py all --dry-run
```

The three variants are `execution_modified_rates` (default), `fixed_individual`,
and `fixed_rates`, using seeds 1–20 and simulator seed 42. Sources are
`experiments/mountaincar_continuous_<variant>/checkpoints/cp.2000.<seed>.0.rslt`.
The runner preserves the full population, inherited instructions, constants,
state flags, and mutation settings. Register bounds match the saved population;
new vector/matrix instructions are disabled. Pendulum uses three observations,
torque in [-2, 2], 200 steps per episode, and 20 training episodes. Generated
transfer configs disable validation and testing (both scheduling and episode
budgets), including Mountain Car training when `--fresh` is used.

Outputs go to
`experiments/mountaincar_continuous_to_pendulum_<variant>/seed_<seed>/pendulum`.
Sources are copied and hashed without modifying the original experiments.
Rerunning resumes the latest complete checkpoint and skips completed runs;
changed configs or simulator seeds require a different `--output-dir`.
`--seed`, `--seeds`, `--jobs`, `--processes`, and `--source-dir` work like the
Pendulum-to-Acrobot runner. `--generations` is the absolute final generation
(default 3000), so `--generations 2500` gives 500 Pendulum generations.
`--fresh` instead trains new scalar-only MountainCarContinuous populations through
generation 2000 before transferring.

On Slurm, submit from the repository root with the normal build/run module
environment and the generation-2000 source checkpoints installed:

```bash
export TPG="$PWD"
cmake --build build --config Release
sbatch scripts/run/transfer-mountaincar-pendulum.slurm execution_modified_rates
sbatch scripts/run/transfer-mountaincar-pendulum.slurm fixed_individual
sbatch scripts/run/transfer-mountaincar-pendulum.slurm fixed_rates
```

Each submission runs seeds 1–20, four at once, with 21 MPI tasks per seed,
6 GB per CPU, and a 23-hour limit using `def-skelly`. Python launches once per
seed and starts MPI with `srun`. Resubmit after a timeout to resume; finished
seeds are skipped. Slurm logs are named
`transfer-mountaincar-pendulum-<array-job-id>_<seed>.out` and `.err`.
An optional second script argument sets the absolute final generation. Additional
arguments after it are forwarded to the Python runner, for example
`3000 --fresh --output-dir "$TPG/experiments/training_only_v1"`.

Runner verification:

```bash
python3 -m unittest discover -s tests -p 'test*transfer_runner.py'
```

### MountainCarContinuous to Acrobot transfer

Take the existing full-population MountainCarContinuous checkpoint at generation
2000 and train Acrobot for **5,000 additional generations**, numbered 2001–7000.
Mountain Car is not retrained by default. The first Acrobot generation evaluates
the restored population before reproduction.

```bash
cmake --build build --config Release
python3 scripts/run/transfer_mountaincar_to_acrobot.py execution_modified_rates
# Validate all 60 source checkpoints without launching training:
python3 scripts/run/transfer_mountaincar_to_acrobot.py all --dry-run
```

This uses the same Mountain Car source experiments, three mutation variants,
seeds 1–20, simulator seed 42, population preservation, scalar mutation settings,
and resume safeguards as the MountainCarContinuous-to-Pendulum runner.
Acrobot uses four observations, torque in [-1, 1], 500 steps per episode,
20 training episodes, and no validation or test episodes. Generated configs also
disable validation and testing for fresh Mountain Car source training. Outputs go to
`experiments/mountaincar_continuous_to_acrobot_<variant>/seed_<seed>/acrobot`.

For Slurm, use the normal build/run module environment and submit from the
repository root:

```bash
export TPG="$PWD"
cmake --build build --config Release
sbatch scripts/run/transfer-mountaincar-acrobot.slurm execution_modified_rates
sbatch scripts/run/transfer-mountaincar-acrobot.slurm fixed_individual
sbatch scripts/run/transfer-mountaincar-acrobot.slurm fixed_rates
```

Each submission runs seeds 1–20, four at once, with 21 MPI tasks per seed,
6 GB per CPU, and a 23-hour limit using `def-skelly`. Resubmit after a timeout
to resume; completed seeds are skipped. Slurm logs are named
`transfer-mountaincar-acrobot-<array-job-id>_<seed>.out` and `.err`.
`--generations` (or the optional second Slurm script argument) is the absolute
final generation, default 7000; setting it to 5000 gives 3,000 Acrobot generations.
The runner also supports `--seed`, `--seeds`, `--jobs`, `--processes`,
`--source-dir`, `--output-dir`, and the explicit `--fresh` option for new sources.
Verify with the same `test*transfer_runner.py` unittest command above.

To redo both Mountain Car transfer pipelines without any validation history,
copy the updated source to the cluster, load the normal build/run modules, and
submit fresh runs into a new output directory:

```bash
export TPG="$PWD"
cmake --build build --config Release
for variant in execution_modified_rates fixed_individual fixed_rates; do
  sbatch scripts/run/transfer-mountaincar-acrobot.slurm "$variant" 7000 \
    --fresh --output-dir "$TPG/experiments/training_only_v1"
  sbatch scripts/run/transfer-mountaincar-pendulum.slurm "$variant" 3000 \
    --fresh --output-dir "$TPG/experiments/training_only_v1"
done
```

Each pipeline trains its own fresh scalar-only Mountain Car population through
generation 2000 before transferring. Acrobot then runs 2001–7000; Pendulum runs
2001–3000. This does not reuse the old source populations or their inherited
vector/matrix instructions. Use a previously unused output directory for a new
attempt; resubmitting with that same directory resumes the new attempt. Omitting
`--fresh` instead reuses the existing Mountain Car source checkpoint, including
any history of validation affecting its evolution. Historical result files are
not modified by these runners.

### Test replay of standalone Acrobot agents

For a local run inside the Linux development container, use:

```bash
bash scripts/run/acrobot-test-replay.sh
```

This rebuilds `TPGExperimentMPI`, prepares the selections, runs and collects
the test episodes, and generates the Acrobot plot. Rebuilding is needed for
the executable to use Acrobot's configurable episode and timestep limits.
The script stops immediately if building or replaying fails.

The replay workflow evaluates the training champion in each run's final
available checkpoint for `acrobot_execution_modified_rates`,
`acrobot_fixed_individual`, and `acrobot_fixed_rates`. It includes all 20 seeds
per variant, including runs that ended early, and records their actual
generation and team ID in the manifest and score CSVs. Each agent receives
20 test episodes with a 500-step limit and the separate test seed offset.

On the Alliance server, use the module environment used to build the executable
and submit from the repository root:

```bash
export TPG="$PWD"
python3 scripts/replay_test_table.py prepare --environment Acrobot
sbatch scripts/run/acrobot-test-replay.slurm
```

The array contains 60 jobs, runs at most eight at once, and uses one MPI process
per replay. Prepare the manifest before submitting and keep it unchanged while
the array runs. Completed JSON results are skipped on resubmission.

After all array jobs finish:

```bash
python3 scripts/replay_test_table.py collect --environment Acrobot --episodes 20
python3 scripts/plot/plot_test_boxplots.py --environment Acrobot
```

Results are saved under `output/test_replay/`:

- `acrobot_test_seed_scores.csv`: one row per run, with test mean, median,
  standard deviation, generation, and team ID.
- `acrobot_test_episode_scores.csv`: the 20 individual episode returns per run.
- `acrobot_test_table_summary.csv`: a summary for each mutation variant.
- `acrobot_test_score_boxplots.pdf` and `.png`: the standalone Acrobot plot.
- `jobs/` and `logs/`: resumable replay results and evaluator stdout/stderr.

The collector recovers individual episode returns from the classic-control
replay evaluator's cumulative reward output. Acrobot score CSVs are written
separately so collecting them preserves the existing tasks' score files.
Copy the `acrobot_test_*` files back to the local `output/test_replay/` directory.
Running `python3 scripts/plot/plot_test_boxplots.py` without an environment filter
combines Acrobot with any other available test results.

### Dynamic Acrobot (three mutation variants)

`configs/acrobot_dynamic_{fixed_rates,fixed_individual,execution_modified_rates}.yaml`
train fresh populations with first-link mass sampled uniformly from [0.5, 1.5).
One mass applies to generations 0–99, a new draw to 100–199, and so on.
All agents, episodes, MPI ranks, and three variants share the same schedule
(`acrobot_mass_seed: 42`), including across evolutionary seeds. Other physical
parameters remain at their existing values. The schedule is determined by the
absolute generation, so checkpoint replay uses the corresponding mass.
The main training log records `acrobot_mass generation ... phase ... m1 ...`.

These configs retain the existing 5000 final generation, 1000 population,
200 offspring, 20 training episodes, and 500 steps per episode. They disable
cached outcomes and the lifetime episode cap: every candidate is reevaluated
each generation so fitness never mixes different masses. This costs more than
the static capped experiments. Validation and test remain disabled as in the
source configs. The mass is not added to policy observations.

Update the remote checkout with these changes and rebuild in your usual cluster
module environment before submitting:

```bash
cmake --build build --config Release --target TPGExperimentMPI
sbatch scripts/run/acrobot-dynamic.slurm
```

The Slurm array runs seeds 1–20 for each variant (60 jobs, at most four concurrent,
21 MPI tasks per job). Adjust account/time/resources in the Slurm header or with
`sbatch` flags for your cluster. Results go under
`experiments/acrobot_dynamic/<variant>/seed_<seed>/`, with a copied config,
launch settings, logs, and checkpoints. Existing run directories are rejected;
use a new output root as the second Slurm-script argument for a new experiment.

To submit through SSH from your local checkout (replace host and remote path):

```bash
bash scripts/run/acrobot-dynamic-ssh.sh --dry-run user@cluster /absolute/remote/LGP
bash scripts/run/acrobot-dynamic-ssh.sh user@cluster /absolute/remote/LGP 5000
```

The SSH wrapper submits the array; it does not copy source or build remotely.
The remote Bash login environment must load the modules required by the build.
