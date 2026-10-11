#!/usr/bin/env python3
"""Extract the best agent at a generation, removing dead/overwritten writes.

Example:
  python3 scripts/extract_generation_agent_instructions.py \
    --environment pendulum --variant execution_modified_rates_no_limit \
    --generation 3960 --seed 8
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

from extract_best_agent_instructions import (
    CHECKPOINT_RE, load_instruction_definitions, parse_checkpoint, reachable_programs,
)

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION_RE = re.compile(r'(?:\d+_){8}')


def load_parameters(config):
    return {key: int(value) for key, value in re.findall(
        r'^\s*(\w+):\s*([+-]?\d+)\s*(?:#.*)?$', config.read_text(), re.MULTILINE)}


def select_agent(directory, generation, seed=None):
    candidates = []
    for log in sorted((directory / 'logs/selection').glob('selection.*.0.csv')):
        log_seed = int(log.name.split('.')[1])
        if seed is not None and seed != log_seed:
            continue
        with log.open(newline='') as handle:
            for row in csv.DictReader(handle):
                if int(row['generation']) == generation and math.isfinite(float(row['best_fitness'])):
                    candidates.append((row, log_seed, log))
    if not candidates:
        raise ValueError(f'No training selection row at generation {generation}, seed {seed}, in {directory}')
    return min(candidates, key=lambda item: (-float(item[0]['best_fitness']), item[1]))


def select_checkpoint(directory, generation, seed, team, exact_only=False):
    candidates = []
    for path in (directory / 'checkpoints').glob(f'cp.*.{seed}.0.rslt'):
        match = CHECKPOINT_RE.fullmatch(path.name)
        if match:
            g = int(match['generation'])
            if g == generation or (not exact_only and g > generation):
                candidates.append((g, path))
    for g, path in sorted(candidates):
        # Avoid parsing large checkpoints that do not contain the selected team.
        with path.open() as handle:
            present = any(line.startswith(f'team:{team}:') for line in handle)
        if present:
            teams, programs = parse_checkpoint(path)
            return g, path, teams, programs
    raise ValueError(f'Team {team} has no usable saved training checkpoint at/after generation {generation}. '
                     'Its instructions cannot be reconstructed from selection logs alone.')


def output_registers(parameters, slots, self_modifying):
    if parameters.get('use_all_scalar_registers', 0):
        count = parameters.get('environment_action_space', parameters.get('min_memory_slots', slots))
        outputs = {('S', i) for i in range(min(slots, max(1, count)))}
        if parameters.get('matrix_modulation', 0):
            outputs.add(('M', 0))
    else:
        outputs = {('S', 0), ({0: 'S', 1: 'S', 2: 'V', 3: 'M'}[parameters.get('continuous_output', 0)], 1 % slots)}
    if self_modifying:
        outputs.update(('S', i) for i in range(2, min(6, slots)))
    return outputs


def analyze(instructions, slots, definitions, outputs, stateful, self_modifying, additive=False):
    """Backward liveness, including loop-carried state and implicit reads.

    Register-level analysis is conservative for vectors/matrices. Random draws
    remain in order to preserve the RNG stream, even when their value is dead.
    """
    decoded = []
    for serialized in instructions:
        a, b, out, opcode, x, y, _, _ = map(int, serialized.rstrip('_').split('_'))
        name, sig = definitions[opcode]
        dst = (sig[0], out % slots)
        reads = set()
        if not name.endswith('CONST_SET_OP'):
            for j, (source, index) in enumerate(((a, x), (b, y))):
                if j + 1 < len(sig) and source == 0:
                    reg = (sig[j + 1], index % slots)
                    if not (self_modifying and reg == ('S', 6)):
                        reads.add(reg)
        if additive or name in ('SCALAR_ACCUM_OP', 'VECTOR_ACCUM_OP', 'SCALAR_CONDITIONAL_OP'):
            reads.add(dst)
        if name == 'OBS_BUFF_SLICE_OP' and a == 0:
            reads.update((sig[1], i) for i in range(slots))
        random = 'UNIFORM_SET_OP' in name or 'GAUSSIAN_SET_OP' in name
        decoded.append((dst, reads, random))

    def backward(end_live):
        live = set(end_live)
        retained = []
        for i in reversed(range(len(decoded))):
            dst, reads, random = decoded[i]
            if self_modifying and dst == ('S', 6):
                continue
            if dst in live or random:
                retained.append(i)
                live.discard(dst)
                live.update(reads)
        return live, sorted(retained)

    end_live = set(outputs)
    while True:
        entry_live, retained = backward(end_live)
        updated = outputs | entry_live if stateful else set(outputs)
        if updated == end_live:
            return retained
        end_live = updated


def render(serialized, slots, definitions, observation_dimension=None, self_modifying=False):
    a, b, out, opcode, x, y, z, _ = map(int, serialized.rstrip('_').split('_'))
    name, sig = definitions[opcode]
    dst = f'{sig[0]}{out % slots}'
    def operand(j, source, index):
        kind = sig[j + 1]
        if source == 1:
            if kind == 'S':
                idx = str(index % observation_dimension) if observation_dimension else f'{index} % observation_dimension'
                return f'observation[{idx}]'
            return f'observation_buffer[{index}] ({kind})'
        if self_modifying and kind == 'S' and index % slots == 6:
            return '0 (S6 decoy read)'
        return f'{"C" if source == 2 else ""}{kind}{index % slots}'
    args = [operand(j, source, index) for j, (source, index) in enumerate(((a, x), (b, y))) if j + 1 < len(sig)]
    if name.endswith('CONST_SET_OP'):
        return f'{dst} = C{sig[0]}{x % slots}'
    if name in ('SCALAR_ACCUM_OP', 'VECTOR_ACCUM_OP'):
        return f'{dst} += {args[0]}'
    if name == 'SCALAR_CONDITIONAL_OP':
        return f'if {args[0]} < {args[1]}: {dst} = -{dst}'
    if name == 'SCALAR_VECTOR_ASSIGN_OP':
        return f'{dst} = {args[0]}[component {y}]'
    if name == 'SCALAR_MATRIX_ASSIGN_OP':
        return f'{dst} = {args[0]}[row {y}, column {z}]'
    return f'{dst} = {name}({", ".join(args)})'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--environment', required=True, help='Experiment environment prefix, e.g. pendulum, ant')
    parser.add_argument('--variant', required=True, help='Experiment suffix, e.g. fixed_rates')
    parser.add_argument('--generation', required=True, type=int)
    parser.add_argument('--seed', type=int, help='Default: highest fitness across seeds at this generation')
    parser.add_argument('--experiments-dir', type=Path, default=ROOT / 'experiments')
    parser.add_argument('--config', type=Path, help='Override experiment YAML path')
    parser.add_argument('--observation-dimension', type=int, help='Resolve scalar observation indices; inferred for native classic control')
    parser.add_argument('--exact-checkpoint', action='store_true', help='Require checkpoint from the requested generation')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        if args.generation < 0 or (args.observation_dimension is not None and args.observation_dimension < 1):
            raise ValueError('Generation must be nonnegative; observation dimension must be positive')
        name = f'{args.environment}_{args.variant}'
        directory = args.experiments_dir / name
        config = args.config
        if config is None:
            config = next((p for p in (ROOT / 'configs' / f'{name}.yaml', ROOT / 'configs/done' / f'{name}.yaml') if p.is_file()), None)
        if config is None:
            raise ValueError('No matching configuration found; pass --config')
        parameters = load_parameters(config)
        if parameters.get('dynamic_lgp', 0) or parameters.get('matrix_modulation', 0):
            raise ValueError('Coordination/modulation effects require additional analysis; this script supports ordinary RegisterMachine execution')
        row, seed, log = select_agent(directory, args.generation, args.seed)
        team = int(row['team_id'])
        checkpoint_generation, checkpoint, teams, programs = select_checkpoint(directory, args.generation, seed, team, args.exact_checkpoint)
        definitions = load_instruction_definitions()
        obs_dim = args.observation_dimension or {'pendulum': 3, 'acrobot': 6, 'mountaincar_continuous': 2}.get(args.environment)
        report = [f'Environment: {args.environment}; variant: {args.variant}',
                  f'Generation: {args.generation}; seed: {seed}; fitness: {row["best_fitness"]}; root team: {team}',
                  f'Selection log: {log}', f'Configuration: {config}', f'Checkpoint: {checkpoint}',
                  'Indices refer to the original instruction sequence.',
                  'Preserves action/bid outputs, mutation-rate outputs, accumulation, conditional updates, and state carried between executions.',
                  'Random draws are retained; vector/matrix analysis is conservative at register granularity.']
        if checkpoint_generation != args.generation:
            report.append(f'NOTE: requested checkpoint was not retained; using saved generation {checkpoint_generation} containing this team. Constants/state are from that saved checkpoint.')
        total = kept = 0
        for owner, pid, program in reachable_programs(team, teams, programs):
            if program is None:
                raise ValueError(f'Missing program {pid} referenced by team {owner}')
            if int(program['action']) >= 0 and int(program['action']) not in teams:
                raise ValueError(f'Missing subteam {program["action"]}')
            instructions = [s for s in program['instructions'] if INSTRUCTION_RE.fullmatch(s)]
            if not instructions:
                raise ValueError(f'Program {pid} has no recognizable instructions')
            slots = int(program['memory_slots'])
            flags = program['stateful_flags'].split('|')[1:]
            stateful = bool(int(program['stateful'])) and all('0' not in flag for flag in flags)
            sm_field = next((s for s in program['instructions'] if s in ('SM0', 'SM1')), None)
            sm = sm_field == 'SM1' if sm_field else bool(parameters.get('self_modifying', 0))
            selected = analyze(instructions, slots, definitions, output_registers(parameters, slots, sm), stateful, sm, bool(parameters.get('operations_additive', 0)))
            total += len(instructions)
            kept += len(selected)
            report += ['', f'Team {owner} | Program {pid} | action {program["action"]} | carries state: {stateful}',
                       f'Original: {len(instructions)}; retained: {len(selected)}; removed: {len(instructions)-len(selected)}',
                       'Constants:', *program['constants'], 'Instructions:']
            for i in selected:
                text = render(instructions[i], slots, definitions, obs_dim, sm)
                if parameters.get('operations_additive', 0):
                    text += '  [then add the previous destination value: operations_additive=1]'
                report.append(f'{i:3}: {text}')
            report.append('Removed indices: ' + ', '.join(str(i) for i in range(len(instructions)) if i not in set(selected)))
        output = args.output or directory / f'agent_instructions.g{args.generation}.seed{seed}.no_overwritten.txt'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('\n'.join(report) + '\n')
        print(f'Generation {args.generation}, seed {seed}, team {team}, fitness {row["best_fitness"]}')
        print(f'{total} -> {kept} instructions; removed {total-kept}')
        print(f'Wrote {output}')
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
