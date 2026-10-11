#!/usr/bin/env python3
"""One-execution structural observation-to-rate dependency metrics for LGP.

Size counts unique instruction nodes on observation-to-output paths. Depth
counts instruction nodes on the longest such path. Paths counts operand-edge
routes (using the same value twice supplies two routes). These are structural
dependencies, not numerical sensitivity or algebraic simplification.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from extract_generation_agent_instructions import (
    ROOT, INSTRUCTION_RE, load_parameters, select_agent, select_checkpoint,
    load_instruction_definitions, reachable_programs, render,
    output_registers,
)
from structural_dependency_graph import build_component_graph, engine_effective_indices

ROLES = {2: 'swap', 3: 'delete', 4: 'add', 5: 'point_mutation', 6: 'decoy'}


def observation_dimension_from_config(config, override=None):
    if override is not None:
        return override
    match = re.search(r'^\s*n_input:\s*[\"\']?(\d+)[\"\']?\s*(?:#.*)?$', config.read_text(), re.MULTILINE)
    if not match:
        raise ValueError('Cannot resolve n_input from config; pass --observation-dimension')
    return int(match[1])


def build_graph(instructions, slots, definitions, observation_dimension,
                self_modifying=True, additive=False, hypothetical_decoy=False,
                stateful=True):
    """SSA-style DAG: every instruction creates a new destination version.

    Incoming state is a separate source, never attributed to this execution's
    observations. Hypothetical decoy enables S6 writes and implicit destination
    reads; ordinary operand reads of S6 remain suppressed, as in the engine.
    """
    nodes = {}
    def source(key, **metadata):
        nodes.setdefault(key, dict(id=key, parents=[], **metadata))
        return key
    versions = {}
    for i in range(slots):
        versions[i] = source(f'initial:S{i}', kind='initial', register=f'S{i}',
                             origin='prior_state' if stateful else 'reset_value')
    for obs in range(observation_dimension):
        source(f'obs:{obs}', kind='observation', observation=obs)
    for i, serialized in enumerate(instructions):
        a, b, out, opcode, x, y, _, _ = map(int, serialized.rstrip('_').split('_'))
        name, sig = definitions[opcode]
        if any(kind != 'S' for kind in sig):
            raise ValueError(f'Instruction {i} uses vector/matrix operations; exact per-observation metrics currently require scalar-only programs')
        dst = out % slots
        if self_modifying and dst == 6 and not hypothetical_decoy:
            continue
        parents = []
        if name.endswith('CONST_SET_OP'):
            parents.append(source(f'constant:CS{x % slots}', kind='constant', register=f'CS{x % slots}'))
        else:
            for j, (src, idx) in enumerate(((a, x), (b, y))):
                if j + 1 >= len(sig):
                    continue
                if src == 1:
                    parents.append(f'obs:{idx % observation_dimension}')
                elif self_modifying and idx % slots == 6:
                    parents.append(source('decoy_zero', kind='constant', value=0))
                elif src == 2:
                    parents.append(source(f'constant:CS{idx % slots}', kind='constant', register=f'CS{idx % slots}'))
                elif src == 0:
                    parents.append(versions[idx % slots])
                else:
                    raise ValueError(f'Unknown source {src} in instruction {i}')
        if name in ('SCALAR_ACCUM_OP', 'SCALAR_CONDITIONAL_OP'):
            parents.append(versions[dst])
        if additive:
            # This is an additional implicit read, even for ACCUM.
            parents.append(versions[dst])
        key = f'instruction:{i}'
        nodes[key] = dict(id=key, kind='instruction', index=i, register=f'S{dst}',
                          operation=name, parents=parents,
                          text=render(serialized, slots, definitions, observation_dimension, self_modifying))
        versions[dst] = key
    return nodes, versions


def metrics(nodes, output, observation_dimension):
    # Python integers preserve exact path counts even for heavily branching DAGs.
    information = {}
    initial_dependencies = {}
    for key, node in nodes.items():
        values = {}
        initial = {node['register']} if node['kind'] == 'initial' else set()
        if node['kind'] == 'observation':
            values[node['observation']] = (set(), 0, 1)
        for parent in node['parents']:
            initial.update(initial_dependencies[parent])
            for obs, (size_set, depth, paths) in information[parent].items():
                previous_set, previous_depth, previous_paths = values.get(obs, (set(), -1, 0))
                values[obs] = (previous_set | size_set, max(previous_depth, depth), previous_paths + paths)
        if node['kind'] == 'instruction':
            values = {obs: (indices | {node['index']}, depth + 1, paths)
                      for obs, (indices, depth, paths) in values.items()}
        information[key] = values
        initial_dependencies[key] = initial
    final = information[output]
    rows = []
    for obs in range(observation_dimension):
        indices, depth, paths = final.get(obs, (set(), 0, 0))
        rows.append(dict(observation=str(obs), size=len(indices), depth=depth, paths=paths,
                         instruction_indices=sorted(indices)))
    union = set().union(*(value[0] for value in final.values())) if final else set()
    rows.append(dict(observation='all', size=len(union),
                     depth=max((v[1] for v in final.values()), default=0),
                     paths=sum(v[2] for v in final.values()), instruction_indices=sorted(union)))
    return rows, sorted(initial_dependencies[output])


def register_metrics(nodes, outputs, slots, source_labels=None):
    """Track each register version independently to each final register.

    Incoming values and values written during execution are separate scopes.
    Source writes are excluded from depth/size: they are the starting value.
    Aggregating written versions unions downstream instructions and sums routes
    per version; it does not claim those versions are independent variables.
    """
    # Disconnected nodes cannot contribute to these destinations. Prune them
    # before propagating per-version paths to keep component graphs economical.
    ancestors, pending = set(), list(outputs.values())
    while pending:
        key = pending.pop()
        if key not in ancestors:
            ancestors.add(key)
            pending.extend(nodes[key]['parents'])
    nodes = {key: value for key, value in nodes.items() if key in ancestors}
    grouped = {}
    for key, node in nodes.items():
        if node['kind'] in ('initial', 'instruction'):
            scope = 'incoming_value' if node['kind'] == 'initial' else 'written_values'
            grouped.setdefault((scope, node['register']), []).append(key)
    propagated = {}
    for keys in grouped.values():
        for source in keys:
            values = {}
            for key, node in nodes.items():
                if key == source:
                    values[key] = (set(), 0, 1)
                    continue
                indices, depth, paths = set(), -1, 0
                for parent in node['parents']:
                    if parent in values:
                        parent_indices, parent_depth, parent_paths = values[parent]
                        indices.update(parent_indices)
                        depth = max(depth, parent_depth)
                        paths += parent_paths
                if paths:
                    if node['kind'] == 'instruction':
                        indices.add(node['index'])
                        depth += 1
                    values[key] = (indices, depth, paths)
            propagated[source] = values
    rows = []
    for scope in ('incoming_value', 'written_values'):
        for label in (source_labels or [f'S{register}' for register in range(slots)]):
            sources = grouped.get((scope, label), [])
            for destination, output in outputs.items():
                indices, depth, paths = set(), 0, 0
                connected = []
                for source in sources:
                    value = propagated[source].get(output)
                    if value is None:
                        continue
                    source_indices, source_depth, source_paths = value
                    indices.update(source_indices)
                    depth = max(depth, source_depth)
                    paths += source_paths
                    connected.append(source)
                rows.append(dict(source_scope=scope, source_register=label,
                    output=f'S{destination}', size=len(indices), depth=depth, paths=paths,
                    identity_paths=int(output in sources),
                    source_nodes=connected, instruction_indices=sorted(indices)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--generation', type=int, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--experiments-dir', type=Path, default=ROOT / 'experiments')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--observation-dimension', type=int)
    parser.add_argument('--exact-checkpoint', action='store_true')
    parser.add_argument('--hypothetical-decoy', action='store_true', help='Also analyze hypothetical S6 writes')
    parser.add_argument('--rate-registers-only', action='store_true', help='Limit register-to-register destinations to S2-S6')
    parser.add_argument('--output-prefix', type=Path, help='Creates PREFIX.csv, PREFIX.registers.csv and PREFIX.json')
    args = parser.parse_args()
    try:
        name = f'{args.environment}_{args.variant}'
        directory = args.experiments_dir / name
        config = args.config or next((p for p in (ROOT / 'configs' / f'{name}.yaml', ROOT / 'configs/done' / f'{name}.yaml') if p.is_file()), None)
        if config is None:
            raise ValueError('No matching configuration; pass --config')
        params = load_parameters(config)
        if params.get('dynamic_lgp', 0) or params.get('matrix_modulation', 0):
            raise ValueError('Coordination/modulation execution is not supported')
        obs_dim = observation_dimension_from_config(config, args.observation_dimension)
        if obs_dim is None or obs_dim < 1 or args.generation < 0:
            raise ValueError('Provide a positive --observation-dimension and nonnegative generation')
        row, seed, log = select_agent(directory, args.generation, args.seed)
        team = int(row['team_id'])
        cp_gen, cp, teams, programs = select_checkpoint(directory, args.generation, seed, team, args.exact_checkpoint)
        definitions = load_instruction_definitions()
        results = []
        register_results = []
        graphs = []
        for owner, pid, program in reachable_programs(team, teams, programs):
            if program is None:
                raise ValueError(f'Missing program {pid}')
            slots = int(program['memory_slots'])
            if slots < 7:
                raise ValueError(f'Program {pid} has fewer than seven scalar registers')
            instructions = [s for s in program['instructions'] if INSTRUCTION_RE.fullmatch(s)]
            flags = program['stateful_flags'].split('|')[1:]
            stateful = bool(int(program['stateful'])) and all('0' not in flag for flag in flags)
            sm_field = next((s for s in program['instructions'] if s in ('SM0', 'SM1')), None)
            sm = sm_field == 'SM1' if sm_field else bool(params.get('self_modifying', 0))
            roots = output_registers(params, slots, sm)
            effective = engine_effective_indices(instructions, slots, definitions, roots, sm)
            modes = ('runtime', 'hypothetical_decoy') if args.hypothetical_decoy else ('runtime',)
            for mode in modes:
                selected = effective if params.get('skip_introns', 1) else list(range(len(instructions)))
                if mode == 'hypothetical_decoy':
                    selected = None
                nodes, outputs = build_component_graph(instructions, slots, definitions, obs_dim,
                    memory_size=int(program['memory_size']), observation_index=int(program['observation_index']),
                    observation_buffer=int(program['observation_buffer']), self_modifying=sm,
                    additive=bool(params.get('operations_additive', 0)), stateful=stateful,
                    selected=selected, hypothetical_decoy=mode == 'hypothetical_decoy')
                targets = ROLES if mode == 'runtime' else {6: 'decoy'}
                register_targets = outputs if mode == 'runtime' else {6: outputs[6]}
                if args.rate_registers_only:
                    register_targets = {i: outputs[i] for i in targets}
                source_labels = sorted({n['register'] for n in nodes.values() if n['kind'] in ('initial', 'instruction')},
                                       key=lambda label: ('SVM'.index(label[0]), int(label[1:])))
                for values in register_metrics(nodes, register_targets, slots, source_labels):
                    register_results.append(dict(environment=args.environment, variant=args.variant,
                        generation=args.generation, checkpoint_generation=cp_gen, seed=seed,
                        root_team=team, owner_team=owner, program=pid, mode=mode,
                        carries_state=int(stateful), **values))
                for register, role in targets.items():
                    rows, initial = metrics(nodes, outputs[register], obs_dim)
                    for values in rows:
                        results.append(dict(environment=args.environment, variant=args.variant,
                            generation=args.generation, checkpoint_generation=cp_gen, seed=seed,
                            fitness=float(row['best_fitness']), root_team=team, owner_team=owner,
                            program=pid, mode=mode, output=f'S{register}', role=role,
                            rate_role_active=int(sm), carries_state=int(stateful),
                            **values, initial_register_dependencies=initial))
                graphs.append(dict(program=pid, mode=mode, nodes=list(nodes.values()),
                                   total_instructions=len(instructions), engine_effective_indices=effective,
                                   memory_size=int(program['memory_size']), observation_dimension=obs_dim,
                                   output_nodes={f'S{i}': node for i, node in register_targets.items()}))
        prefix = args.output_prefix or directory / f'observation_rate_dependencies.g{args.generation}.seed{seed}'
        csv_path, json_path = Path(str(prefix) + '.csv'), Path(str(prefix) + '.json')
        register_csv_path = Path(str(prefix) + '.registers.csv')
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            for result in results:
                writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in result.items()})
        with register_csv_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(register_results[0]))
            writer.writeheader()
            for result in register_results:
                writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in result.items()})
        payload = dict(metadata=dict(selection_log=str(log), config=str(config), checkpoint=str(cp),
            requested_generation=args.generation, checkpoint_generation=cp_gen,
            execution_scope='one execution; incoming state is separate',
            size='unique instruction nodes on observation-to-output paths',
            depth='maximum instruction-node count along a path; zero if disconnected',
            paths='operand-edge routes; duplicate operand uses count separately; exact integers',
            aggregation='all unions instruction sets, takes max depth, and sums observation path counts',
            hypothetical_decoy='enable S6 writes and implicit destination reads; keep ordinary S6 operand reads zero',
            limitations='structural dependence, not numeric influence; no value/branch simplification, RNG-stream edges, or inter-program control-flow edges; observation buffer must be one',
            graph_granularity='output-component nodes; size counts unique instruction indices; depth counts instruction transitions; reductions have operand edges from each consumed component',
            implementation_notes='matches current engine: VECTOR_RECIPROCAL writes only element 0; scalar component extracts use effective indices 0; scalar observation buffers used by mixed operations remain zero; in-place transpose selected by index equality',
            initial_register_dependencies='prior_state when carries_state=1, otherwise per-execution reset values',
            register_sources='incoming_value: register at execution entry; written_values: all SSA values written to that register during this execution',
            register_targets='final scalar values (S2-S6 only when --rate-registers-only); final S6 in hypothetical view',
            register_size_depth='downstream instruction nodes only; source write is excluded',
            register_paths='sum operand-edge paths per source version; shared routes can be counted from multiple versions of the same source register',
            register_identity='unchanged source equals final value: size=0, depth=0, paths=1; identity_paths marks that route'),
            metrics=results, register_metrics=register_results, graphs=graphs)
        json_path.write_text(json.dumps(payload, indent=2) + '\n')
        print(f'Wrote {csv_path}\nWrote {register_csv_path}\nWrote {json_path}')
        if cp_gen != args.generation:
            print(f'NOTE: saved checkpoint generation {cp_gen} contains the selected generation-{args.generation} team; constants/state are from that checkpoint.')
        for result in results:
            if result['observation'] == 'all':
                print(f'Program {result["program"]} {result["mode"]} {result["output"]} ({result["role"]}): '
                      f'size={result["size"]}, depth={result["depth"]}, paths={result["paths"]}; '
                      f'incoming={result["initial_register_dependencies"]}')
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
