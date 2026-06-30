# MuJoCo environment timing notes

This codebase has **two different notions of "episode length"**:

- **Control steps**: one call to an environment's `sim_step(...)` (one action decision).
- **Simulation steps**: one MuJoCo `mj_step(...)` call (physics integrator step).

MuJoCo tasks advance physics by calling:

- `do_simulation(action, frame_skip_)`

which runs `mj_step(...)` exactly `frame_skip_` times per control step.

## Keeping episode duration consistent when changing `frame_skip_`

MuJoCo tasks in this repo terminate on a **fixed simulation-step budget**:

- `max_sim_steps_ = mj_max_timestep * frame_skip_`

So if you change `frame_skip_`, the episode will run for roughly the **same simulated time**,
but it will contain a different number of **control steps** (action decisions).

### Example: HalfCheetah-v4 style horizon

If you run with `mj_max_timestep: 1000` and `frame_skip_ = 5`, then:

- `max_sim_steps_ = 1000 * 5 = 5000` MuJoCo physics steps

If you later change `frame_skip_` to `1`, the episode will still end after ~5000 physics steps
but will now contain ~5000 control steps.

## Config keys

- `mj_max_timestep`: sets the base horizon used to compute the simulation-step limit


