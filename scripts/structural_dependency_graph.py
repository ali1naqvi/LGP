"""Component-level dataflow for the repository's RegisterMachine implementation.

Instruction nodes represent output components; size still counts unique original
instruction indices. Reductions expose one operand edge per consumed component.
"""


def engine_effective_indices(instructions, slots, definitions, outputs, self_modifying):
    """Mirror MarkIntrons, including its conservative fixed-point expansion."""
    decoded = []
    for raw in instructions:
        a, b, out, op, x, y, _, _ = map(int, raw.rstrip('_').split('_'))
        sig = definitions[op][1]
        reads = []
        for j, (source, index) in enumerate(((a, x), (b, y))):
            if j + 1 < len(sig) and source == 0:
                reg = (sig[j + 1], index % slots)
                if not (self_modifying and reg == ('S', 6)):
                    reads.append(reg)
        decoded.append(((sig[0], out % slots), reads))
    live = set(outputs)
    stateless = set()
    for i in reversed(range(len(decoded))):
        dst, reads = decoded[i]
        if dst in live and not (self_modifying and dst == ('S', 6)):
            stateless.add(i)
            live.update(reg for reg in reads if reg[0] != 'M' or reg[1] == 0)
    for _ in instructions:
        selected = []
        previous = set(live)
        for i, (dst, reads) in enumerate(decoded):
            if (dst in live and not (self_modifying and dst == ('S', 6))) or i in stateless:
                selected.append(i)
                live.update(reads)
        if live == previous:
            return selected
    return selected if instructions else []


def build_component_graph(instructions, slots, definitions, observation_dimension,
                          memory_size=1, observation_index=0, observation_buffer=1,
                          self_modifying=True, additive=False, stateful=True,
                          selected=None, hypothetical_decoy=False):
    if observation_buffer != 1:
        raise ValueError('Component analysis currently requires observation_buffer=1')
    nodes, versions = {}, {}
    def source(key, **metadata):
        nodes.setdefault(key, dict(id=key, parents=[], **metadata))
        return key
    def coords(kind):
        return [(r, c) for r in range(1 if kind == 'S' else memory_size)
                for c in range(memory_size if kind == 'M' else 1)]
    for kind in ('S', 'V', 'M'):
        for slot in range(slots):
            versions[kind, slot] = {}
            for r, c in coords(kind):
                label = f'{kind}{slot}'
                versions[kind, slot][r, c] = source(f'initial:{label}:{r}:{c}',
                    kind='initial', register=label, component=[r, c],
                    origin='prior_state' if stateful else 'reset_value')
    for obs in range(observation_dimension):
        source(f'obs:{obs}', kind='observation', observation=obs)

    enabled = set(range(len(instructions))) if selected is None else set(selected)
    scalar_memory_ops = {'SCALAR_VECTOR_PRODUCT_OP', 'SCALAR_MATRIX_PRODUCT_OP',
                         'SCALAR_BROADCAST_OP'}
    for i, raw in enumerate(instructions):
        if i not in enabled:
            continue
        a, b, out, opcode, x, y, _, _ = map(int, raw.rstrip('_').split('_'))
        name, sig = definitions[opcode]
        kind, slot = sig[0], out % slots
        if self_modifying and (kind, slot) == ('S', 6) and not hypothetical_decoy:
            continue
        old = versions[kind, slot].copy()

        def operand(j):
            src, idx = ((a, x), (b, y))[j]
            typ = sig[j + 1]
            if src == 0:
                if self_modifying and typ == 'S' and idx % slots == 6 and name not in scalar_memory_ops:
                    return {(0, 0): source('decoy_zero', kind='constant', value=0)}
                return versions[typ, idx % slots].copy()
            if src == 2:
                if self_modifying and typ == 'S' and idx % slots == 6 and name not in scalar_memory_ops:
                    return {(0, 0): source('decoy_zero', kind='constant', value=0)}
                return constant(typ, idx)
            if src != 1:
                raise ValueError(f'Invalid input source {src}')
            if typ == 'S':
                # These mixed operations use InputMemory rather than scalar_in*.
                # Run populates vector/matrix observation buffers only.
                if name in scalar_memory_ops or 'UNIFORM_SET' in name or 'GAUSSIAN_SET' in name:
                    return {(0, 0): source('scalar_observation_buffer_zero', kind='constant', value=0)}
                return {(0, 0): f'obs:{idx % observation_dimension}'}
            return {(r, c): f'obs:{(observation_index + r * (memory_size if typ == "M" else 1) + c) % observation_dimension}'
                    for r, c in coords(typ)}

        def constant(typ, idx):
            return {(r, c): source(f'constant:C{typ}{idx % slots}:{r}:{c}',
                    kind='constant', register=f'C{typ}{idx % slots}', component=[r, c])
                    for r, c in coords(typ)}

        args = [operand(j) for j in range(len(sig) - 1)] if not name.endswith('CONST_SET_OP') else []
        parents = {}
        if name.endswith('CONST_SET_OP'):
            parents = {cell: [key] for cell, key in constant(kind, x).items()}
        elif name in ('SCALAR_VECTOR_ASSIGN_OP', 'SCALAR_MATRIX_ASSIGN_OP'):
            # in2IdxE_/in3IdxE_ default to zero; Run only sets effective indices
            # 0 and 1. The serialized component indices are not applied.
            parents[0, 0] = [args[0][0, 0]]
        elif name == 'SCALAR_BROADCAST_OP':
            parents = {cell: [args[0][0, 0]] for cell in coords(kind)}
        elif name in ('SCALAR_VECTOR_PRODUCT_OP', 'SCALAR_MATRIX_PRODUCT_OP'):
            parents = {cell: [args[0][0, 0], args[1][cell]] for cell in coords(kind)}
        elif name in ('VECTOR_NORM_OP', 'MATRIX_NORM_OP', 'VECTOR_MEAN_OP',
                      'MATRIX_MEAN_OP', 'VECTOR_ST_DEV_OP', 'MATRIX_ST_DEV_OP'):
            parents[0, 0] = list(args[0].values())
        elif name == 'VECTOR_INNER_PRODUCT_OP':
            parents[0, 0] = [arg[r, 0] for r in range(memory_size) for arg in args]
        elif name == 'VECTOR_OUTER_PRODUCT_OP':
            parents = {(r, c): [args[0][r, 0], args[1][c, 0]] for r, c in coords(kind)}
        elif name in ('MATRIX_VECTOR_PRODUCT_OP', 'MATRIX_MATRIX_PRODUCT_OP'):
            parents = {(r, c): [key for k in range(memory_size)
                                for key in (args[0][r, k], args[1][k, c])]
                       for r, c in coords(kind)}
        elif name in ('VECTOR_COLUMN_BROADCAST_OP', 'VECTOR_ROW_BROADCAST_OP'):
            parents = {(r, c): [args[0][(r if name == 'VECTOR_COLUMN_BROADCAST_OP' else c), 0]]
                       for r, c in coords(kind)}
        elif name in ('MATRIX_ROW_NORM_OP', 'MATRIX_ROW_MEAN_OP', 'MATRIX_ROW_ST_DEV_OP'):
            parents = {(r, 0): [args[0][r, c] for c in range(memory_size)] for r in range(memory_size)}
        elif name == 'MATRIX_COLUMN_NORM_OP':
            parents = {(c, 0): [args[0][r, c] for r in range(memory_size)] for c in range(memory_size)}
        elif name == 'MATRIX_TRANSPOSE_OP':
            # Current implementation chooses transposeInPlace by index equality,
            # even for a constant or observation operand.
            input_index = x % (observation_buffer if a == 1 else slots)
            actual_input = old if slot == input_index else args[0]
            parents = {(r, c): [actual_input[c, r]] for r, c in coords(kind)}
        elif name == 'VECTOR_RECIPROCAL_OP':
            # This implementation writes only component zero, leaving the rest.
            parents[0, 0] = [args[0][0, 0]]
        elif name == 'OBS_BUFF_SLICE_OP':
            raise ValueError('OBS_BUFF_SLICE_OP requires a history-specific model')
        elif 'UNIFORM_SET_OP' in name or 'GAUSSIAN_SET_OP' in name:
            parents = {cell: [arg[0, 0] for arg in args] for cell in coords(kind)}
        else:
            # Remaining operations are scalar or elementwise.
            if any(set(arg) != set(coords(kind)) for arg in args):
                raise ValueError(f'No component mapping for {name}')
            parents = {cell: [arg[cell] for arg in args] for cell in coords(kind)}
        if name in ('SCALAR_ACCUM_OP', 'VECTOR_ACCUM_OP', 'SCALAR_CONDITIONAL_OP'):
            for cell in parents:
                parents[cell].append(old[cell])
        if additive:
            for cell in coords(kind):
                parents.setdefault(cell, [old[cell]]).append(old[cell])
        updated = old.copy()
        for (r, c), inputs in parents.items():
            key = f'instruction:{i}:{r}:{c}'
            nodes[key] = dict(id=key, kind='instruction', index=i,
                register=f'{kind}{slot}', component=[r, c], operation=name,
                parents=inputs, serialized=raw)
            updated[r, c] = key
        versions[kind, slot] = updated
    return nodes, {slot: versions['S', slot][0, 0] for slot in range(slots)}
