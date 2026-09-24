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
