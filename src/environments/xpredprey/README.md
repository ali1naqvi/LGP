# xpredprey

A native C++ predator–prey robot environment, located with the repository's
other environments under `src/environments`. It was extracted from
[`evorobotpy2`](https://github.com/snolfi/evorobotpy2).

The environment contains two simulated MarXBot robots:

- a predator with 80% of the prey's maximum speed;
- a prey robot;
- calibrated infrared sensors, an omnidirectional camera, ground-gradient,
  time, and bias sensors;
- two wheel-speed actions per robot.

The command-line program runs a single episode with constant wheel actions.
When selected as the LGP task, the environment uses competitive coevolution
with two independent elite populations:

- predator teams reproduce only from predator parents and are ranked by
  capture speed;
- prey teams reproduce only from prey parents and are ranked by survival time;
- a centrally seeded schedule pairs candidates with the opposing population;
- BOTH candidate and opponent appearances retain execution-produced mutation
  registers during training; only candidate appearances contribute fitness;
- the `n_root` and `n_root_gen` budgets are split evenly between the roles
  (the predator receives the extra slot when either value is odd).

## Build with LGP

From the repository root, the normal CMake build includes xpredprey. To build
only its targets:

```sh
cmake -S . -B build
cmake --build build --target xpredprey
./build/src/environments/xpredprey/xpredprey
```

## Standalone build

Requirements:

- CMake 3.20 or newer
- a C++17 compiler

```sh
cd src/environments/xpredprey
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

No Python, Cython, GSL, or other third-party runtime libraries are required.

## Evolve both populations

After building the full LGP project, the CLI discovers
`configs/xpredprey.yaml` as the `xpredprey` environment:

```sh
tpg evolve xpredprey -p 4 -s 42
```

`keep_old_outcomes` is disabled in this configuration because fitness must be
recomputed as the opposing population changes. Population roles are stored in
both disk and MPI checkpoints.

The two populations can independently select the mutation-rate variant in
`configs/xpredprey.yaml`. For example, this compares a self-modifying predator
population against a regular prey population:

```yaml
predator_population:
  self_modifying: 1

prey_population:
  self_modifying: 0
```

Set `self_modifying` to `1` to enable inherited S2-S6 rate registers and their
execution-time modification. **The shipped configuration keeps
`reset_self_modifying_before_mutation: 1`**: encounter outputs persist during
training, but mutation uses the inherited constants rather than those outputs.
Setting the reset flag to `0` instead makes mutation consume the final encounter
outputs. This flag remains global; it is not a per-role setting.
Set `self_modifying` to `0` to use the fixed
`p_instructions_swap`, `p_instructions_delete`, `p_instructions_add`, and
`p_instructions_mutate` and `p_instructions_redundancy` values from
`program_parameters`. The settings affect
initial programs, offspring, crossover, and evaluation. New checkpoints store
the variant on every program so a resumed experiment preserves its original
population definitions; older checkpoints adopt the current config values.

## Encounter history and evaluation isolation

`xpredprey_n_eval_train: E` means E scored candidate appearances per individual,
plus opponent appearances. For equal population sizes, every individual has
exactly E opponent appearances too (2E total). For unequal sizes, opponent
counts differ by at most one **within each role**; equal exposure across roles
is mathematically impossible when their population sizes differ and every
match uses one participant of each role.

The master sorts team IDs, shuffles each population using `seed_aux`, generation,
and phase, and constructs a rotating opponent schedule. Candidate roles alternate
their order by episode. Each encounter has its own logged seed and ordinal.
`seed_with_episode_number` does not override this schedule. The simulator still
uses its existing fixed starting positions; a new encounter seed does not imply
new starting positions.

The complete phase runs on **one evaluator worker** so stateful policy memory
and team learning state follow the encounter order, as do S2-S7. Launching more
MPI ranks does not change the schedule; extra workers remain idle and participate
in the collectives. This reference implementation sacrifices parallel evaluation
speed. A future parallel implementation must synchronize the full runtime state,
not only the mutation registers. At least two MPI processes are required
(one master, one evaluator).

Live S2-S7 outputs are returned to the master after training and included in
both MPI and disk checkpoints. Surviving programs therefore keep their rate
history into the next generation. Offspring start with their inherited constants
after variation. This does not add persistence of ordinary working memory or
Hebbian state across generations/checkpoint reloads.

Validation and testing run on disposable worker copies of the current training
population's rate state. They return scores and logs but **never** write rate
outputs into the training population, even if an unsolicited rate record is
received. They also do not consume the master's training RNG. Their encounter
histories are local to the phase. `validation_mod` and `test_mod` remain disabled
in the supplied config; isolation applies when you enable them. The existing
test procedure compares validation champions, not a frozen cross-treatment
opponent panel.

There is no explicit win/loss observation or extra post-terminal policy execution.
The log associates rate changes with a match result; it does not establish that
losing caused a change. Both participants advance their execution timestep and
reward/learning bookkeeping. S6 controls adjacent instruction duplication. S7 is
the isolated decoy: policy reads and writes are blocked, while inherited S7 can
evolve through constant mutation.

## Logs

Only the master writes these CSVs, avoiding concurrent worker writes:

- `logs/xpredprey/encounters.<seed_tpg>.<pid>.csv`: one row per participant/program
  per encounter. Includes generation, phase (0 train / 1 validation / 2 test),
  encounter ID and seed, episode, team ID, role, opponent ID, candidate/opponent
  appearance, encounter count, program ID, self-modification flag, elapsed steps,
  role reward, capture/timeout result, program/effective lengths, and before/after
  raw S2-S7 values. Also includes inherited raw values and sigmoid-transformed
  before/after probabilities. Counts restart at each generation/phase and include
  both appearances. For multi-program teams the same team count repeats on its
  program rows. Raw values preserve spikes/non-finite values; transformed values
  use the same safety mapping as mutation. For fixed-rate programs these registers
  are ordinary memory, not mutation rates; filter by `self_modifying` accordingly.
- `logs/xpredprey/reproduction.<seed_tpg>.<pid>.csv`: one row per mutated program
  per mutation pass. Includes child team/role, parent/child program IDs, reset flag,
  parent encounter outputs (before clone sanitization), inherited values before/after the
  pass, and the five probabilities actually used for
  swap/delete/add/point-mutation/redundancy.
  These are operator probabilities, not a record of which random operators fired.
  S7 is logged but never controls a mutation operator.

CSV files are append-only for a seed/PID pair and flushed at phase boundaries.
Use distinct run directories or PIDs for independent runs/resume branches.
Encounter IDs are unique only within `(generation, phase)`; program rows also
need team/program/appearance identifiers. Detailed encounter logs can be large
at the default population and generation budgets.

## Engine and MPI regression tests

The standalone simulator tests cover rewards and balanced scheduling. The
additional suite builds the real engine and XPredPrey evaluation protocol without
requiring MuJoCo, rendering libraries, or a downloaded test framework:

```sh
cmake -S tests/xpredprey -B build/xpredprey-tests -DCMAKE_BUILD_TYPE=Release
cmake --build build/xpredprey-tests -j 4
ctest --test-dir build/xpredprey-tests --output-on-failure
```

Requirements: a C++23 compiler, Eigen, yaml-cpp, Boost MPI/serialization/iostreams,
MPI, CURL, and BZip2. Tests use tiny populations and three-step episodes, not a
training experiment. They check candidate/opponent rate continuity, logging,
reset-before mutation, rate wire mapping, live-rate checkpoint round trips,
held-out isolation, role elite quotas, idle-worker participation, and matching
results with 2 versus 4 MPI processes. MPI tests require permission to open local
sockets. To run the non-multiprocess checks only, use `ctest --test-dir
build/xpredprey-tests --output-on-failure -E xpredprey_mpi`.

## Run

From the xpredprey environment directory:

```sh
./build/xpredprey
```

The defaults drive the predator straight toward a stationary prey, which
provides a deterministic setup check. To see every option:

```sh
./build/xpredprey --help
```

For example:

```sh
./build/xpredprey --steps 500 --seed 7 \
  --predator 0.8 1.0 --prey 0.5 0.7
```

Each wheel action is clamped to `[-1, 1]`. The program reports whether the
predator captured the prey, the number of elapsed steps, and the predator's
reward.
