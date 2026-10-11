# Observation-to-mutation-rate dependency data

Run from the repository root:

```bash
python3 scripts/analyze_observation_rate_dependencies.py \
  --environment pendulum \
  --variant execution_modified_rates_no_limit \
  --generation 3960 --seed 8
```

Omitting `--seed` selects the highest training fitness across seeds at the
requested generation. Use `--config PATH` for configurations outside `configs`
or `configs/done`. Observation dimensions are read from the configuration's
`n_input`; `--observation-dimension N` overrides them. In this repository,
MountainCarContinuous and Acrobot each have four observations, Half-Cheetah 17,
Reacher 10, and Pendulum three. The old `analyze_observation_rate_semantics.py`
entry point remains available for compatibility.

The script uses the selected team in an exact training checkpoint if available;
otherwise it searches later saved checkpoints containing that same team and
records both generations. `--exact-checkpoint` disables this fallback. Constants
and state in a later checkpoint are not a reconstruction of the earlier state.

Three files are written next to the experiment's logs, or at `--output-prefix PATH`:

- `.csv`: one row per observation/output pair, plus an `all` row per output.
  Includes environment, variant, requested and saved generation, seed, fitness,
  team, program, view, output, role, size, depth, paths, original instruction
  indices, and incoming-register dependencies.
- `.json`: the same metrics, their definitions, source paths, and the versioned
  dependency graphs. `parents` lists preserve duplicate operand edges.
- `.registers.csv`: scalar/vector/matrix source registers to final scalar destinations, with
  two source scopes: `incoming_value` for the entry value, and `written_values`
  for values written to the source register during execution. The hypothetical
  view targets only S6. JSON also includes these rows in `register_metrics`.
  `--rate-registers-only` limits destinations to S2-S6.

Register-to-register size/depth count downstream instruction nodes, excluding
the source write itself. `written_values` unions these nodes across connected
source versions and sums paths per version. Multiple versions can contribute
overlapping routes; this is an aggregation of value-version relationships,
not a count of independent source variables. `source_nodes` identifies the
connected versions. A source equal to the final value has one zero-length
identity path (size/depth 0, paths 1), marked by `identity_paths`. This records
unchanged state such as runtime S6 without claiming a computation occurred.

## Definitions

Each write creates a new register version, so an overwritten value only remains
connected if it is read before the overwrite. Accumulation reads the previous
destination. Conditional sign changes read both their condition operands and
their previous destination. Additive mode also reads the previous destination.

For a particular observation and final output:

- **Size** is the number of unique instruction nodes lying on paths between
  that observation and the output. Constant-only instructions are excluded.
- **Depth** is the maximum number of instruction nodes on one such path. A
  direct observation-to-write dependency has depth 1; a disconnected pair has 0.
- **Paths** counts operand-edge routes. For example, `S2 = obs[0]` followed by
  `S2 += S2` has size 2, depth 2, and paths 2. Paths use arbitrary-precision
  integers; no floating-point rounding is involved.

The `all` row unions instruction sets, takes the maximum depth, and sums path
counts across observations. Its size is not the sum of individual sizes because
an instruction can depend on several observations.

The analysis covers **one execution**. Incoming register state is a separate
source, not attributed to this execution's observations. The
`initial_register_dependencies` field lists all incoming values contributing to
the output. `carries_state=1` means these are prior execution state; otherwise
they are per-execution reset values. This follows the engine's current behavior:
if the program is globally stateless or any per-register flag is stateless,
ordinary execution resets working memory.

Mutation-rate targets are raw S2 (swap), S3 (delete), S4 (add), and S5 (point
mutation), before probability conversion. S6 is the decoy. When self-modification
is disabled these register labels are only positional; `rate_role_active=0`
marks that they are not active mutation-rate outputs.

The runtime view ignores S6 writes and suppresses ordinary S6 operand reads,
matching self-modifying engine execution. The optional `--hypothetical-decoy` view
enables S6 writes and its implicit accumulator/conditional destination reads,
while still suppressing ordinary operand reads of S6. This isolates its written
computation without introducing feedback from the decoy into mutation rates.

## Interpretation and limits

These are **structural dependency metrics**, not empirical numerical sensitivity.
They do not simplify constants, prove branch conditions, cancel expressions, or
account for numerical clipping/sanitization. A structural dependency may have no
numerical effect on a particular trajectory.

Vector and matrix instructions are modeled by output component. Size counts
unique original instruction indices; paths count component operand routes, so a
matrix reduction can have many paths even with one instruction. The graph starts
with the engine's MarkIntrons execution mask. Reachable programs are reported
separately; inter-program control flow and shared random-number-stream effects
are not graph edges. Rate probability conversion is not counted as an instruction.
Coordination/modulation and observation buffers larger than one are unsupported.

Component mappings follow the current engine, including its implementation
details: VECTOR_RECIPROCAL updates only element zero; component extraction uses
effective indices zero; mixed scalar/vector operations read the unpopulated
scalar observation buffer; transpose chooses its in-place branch by index
equality. These are implementation observations, not changes to engine behavior.

For generation 3960, seed 8, Pendulum execution_modified_rates_no_limit, the saved
checkpoint is generation 3963. The all-observation runtime results are:

| Register | Role | Size | Depth | Paths | Incoming state |
|---|---|---:|---:|---:|---|
| S2 | Swap | 0 | 0 | 0 | None |
| S3 | Delete | 1 | 1 | 1 | None |
| S4 | Add | 29 | 13 | 27 | None |
| S5 | Point mutation | 48 | 18 | 35 | S4 |
| S6 | Decoy | 0 | 0 | 0 | S6 |

There are no S6 writes in this agent; its hypothetical decoy result is also zero.

## Validation

```bash
python3 scripts/test_analyze_observation_rate_semantics.py
python3 scripts/test_extract_generation_agent_instructions.py
python3 scripts/test_structural_dependency_graph.py
```

The semantic tests use known graphs to check unique sizes, longest depths,
duplicate operand paths, shared branches, overwritten values, accumulation,
conditional sign changes, incoming state, additive mode, and decoy behavior.
