#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
from collections import deque
from pathlib import Path

EXPERIMENT_DIR = "experiments/gradient_vector_continuous"
OUTPUT_FILENAME = "best_agents_instructions.txt"
# FastSimGradient has one observation in this no-reward configuration:
# normalised local concentration.  Change this if the selected experiment has
# additional observation channels.
OBSERVATION_DIMENSION = 1


CHECKPOINT_RE = re.compile(r"^cp\.(?P<generation>\d+)\.(?P<seed>\d+)\.(?P<phase>\d+)\.rslt$")
MEMORY_LABELS = {"Scalar": "S", "Vector": "V", "Matrix": "M"}
MEMORY_TYPE_LABELS = {0: "S", 1: "V", 2: "M"}


def load_instruction_definitions() -> dict[int, tuple[str, list[str]]]:
    """Read operation names/signatures from the source, avoiding a duplicate table."""
    root_dir = Path(__file__).resolve().parents[1]
    header = (root_dir / "src" / "engine" / "instruction.h").read_text(encoding="utf-8")
    source = (root_dir / "src" / "engine" / "instruction.cc").read_text(encoding="utf-8")

    opcode_by_symbol = {
        symbol: int(opcode)
        for symbol, opcode in re.findall(r"static const int (\w+)_ = (\d+);", header)
    }
    signatures_by_symbol = {
        symbol: [MEMORY_LABELS.get(memory_type, "?") for memory_type in re.findall(r"MemoryEigen::k(\w+)Type_", values)]
        for symbol, values in re.findall(
            r"op_signatures_\[(\w+)_\]\s*=\s*\{(.*?)\};", source, re.DOTALL
        )
    }
    return {
        opcode: (symbol, signatures_by_symbol.get(symbol, []))
        for symbol, opcode in opcode_by_symbol.items()
    }


def render_instruction(serialized: str, memory_slots: int,
                       observation_dimension: int,
                       definitions: dict[int, tuple[str, list[str]]]) -> str:
    """Convert the compact checkpoint tuple into a readable register operation."""
    values = [int(value) for value in serialized.rstrip("_").split("_")]
    if len(values) != 8:
        return f"<unrecognised instruction: {serialized}>"
    input0_source, input1_source, output_index, opcode, input0_index, input1_index, _, _ = values
    op_name, signature = definitions.get(opcode, (f"UNKNOWN_OPCODE_{opcode}", []))
    if not signature:
        return f"{op_name}  [raw: {serialized}]"

    slot_count = max(memory_slots, 1)
    output = f"{signature[0]}{output_index % slot_count}"

    def operand(input_number: int, source: int, index: int) -> str:
        if input_number >= len(signature) - 1:
            return ""
        memory_type = signature[input_number + 1]
        if source == 1:
            resolved_index = index % max(observation_dimension, 1)
            return f"observation[{resolved_index}] ({memory_type})"
        if source == 2:
            return f"C{memory_type}{index % slot_count}"
        return f"{memory_type}{index % slot_count}"

    inputs = [
        operand(0, input0_source, input0_index),
        operand(1, input1_source, input1_index),
    ]
    inputs = [value for value in inputs if value]
    return f"{output} = {op_name}({', '.join(inputs)})"


def load_program_parameters(experiment_dir: Path) -> dict[str, int]:
    """Read the intron-analysis settings from the matching experiment config."""
    config_path = Path(__file__).resolve().parents[1] / "configs" / f"{experiment_dir.name}.yaml"
    parameters = {
        "continuous_output": 0,
        "environment_action_space": 0,
        "matrix_modulation": 0,
        "min_memory_slots": 1,
        "self_modifying": 0,
        "use_all_scalar_registers": 0,
    }
    if not config_path.is_file():
        return parameters

    for line in config_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*(\w+):\s*([+-]?\d+)", line)
        if match and match[1] in parameters:
            parameters[match[1]] = int(match[2])
    return parameters


SELF_MODIFYING_PROBABILITY_EPSILON = 1e-6
SELF_MODIFYING_RATE_TEMPERATURE = 1.0


def stable_sigmoid(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def raw_tendency_to_probability(value: float) -> float:
    return (
        SELF_MODIFYING_PROBABILITY_EPSILON
        + (1.0 - 2.0 * SELF_MODIFYING_PROBABILITY_EPSILON)
        * stable_sigmoid(value / SELF_MODIFYING_RATE_TEMPERATURE)
    )


def format_scalar_constant(
    label: str, slot: int, raw_value: str, self_modifying: bool,
) -> str:
    try:
        numeric = float(raw_value)
    except ValueError:
        return f"{label}{slot}={raw_value}"
    if self_modifying and label == "CS" and 1 <= slot <= 4:
        probability = raw_tendency_to_probability(numeric)
        return f"{label}{slot}={raw_value} (p={probability:.6f})"
    return f"{label}{slot}={raw_value}"


def parse_memory_constants(
    fields: list[str], self_modifying: bool = False,
) -> list[str]:
    """Return compact register constants from a MemoryEigen checkpoint record."""
    try:
        memory_type = int(fields[2])
        memory_slots = int(fields[3])
        memory_size = int(fields[4])
    except (IndexError, ValueError):
        return []

    label = f"C{MEMORY_TYPE_LABELS.get(memory_type, f'T{fields[2]}')}"
    values = fields[5:]
    constants: list[str] = []
    cursor = 0
    for slot in range(memory_slots):
        if memory_type == 0:
            width = 1
        elif memory_type == 1:
            width = memory_size
        else:
            width = memory_size * memory_size

        register_values = values[cursor:cursor + width]
        cursor += width
        if len(register_values) < width:
            constants.append(f"{label}{slot}=<missing>")
        elif memory_type == 0:
            constants.append(
                format_scalar_constant(label, slot, register_values[0], self_modifying)
            )
        elif memory_type == 1:
            constants.append(f"{label}{slot}=[{', '.join(register_values)}]")
        else:
            rows = [
                "[" + ", ".join(register_values[row:row + memory_size]) + "]"
                for row in range(0, width, memory_size)
            ]
            constants.append(f"{label}{slot}=[{', '.join(rows)}]")
    return constants


def effective_instruction_indices(
    instructions: list[str], memory_slots: int,
    definitions: dict[int, tuple[str, list[str]]], program_parameters: dict[str, int],
) -> list[int]:
    """Return the instruction indices retained by RegisterMachine::MarkIntrons."""
    slot_count = max(memory_slots, 1)
    decoded: list[tuple[tuple[str, int], list[tuple[str, int]]]] = []
    for serialized in instructions:
        values = [int(value) for value in serialized.rstrip("_").split("_")]
        if len(values) != 8:
            # Preserve malformed instructions rather than silently dropping them.
            decoded.append((("?", 0), []))
            continue
        input0_source, input1_source, output_index, opcode, input0_index, input1_index, _, _ = values
        _, signature = definitions.get(opcode, ("", []))
        if not signature:
            decoded.append((("?", 0), []))
            continue
        inputs = []
        for input_number, (source, index) in enumerate(
            ((input0_source, input0_index), (input1_source, input1_index))
        ):
            if input_number + 1 < len(signature) and source == 0:
                inputs.append((signature[input_number + 1], index % slot_count))
        decoded.append(((signature[0], output_index % slot_count), inputs))

    effective = {("S", 0)}
    if program_parameters["use_all_scalar_registers"]:
        action_space = program_parameters["environment_action_space"] or program_parameters["min_memory_slots"]
        effective = {("S", index) for index in range(min(max(action_space, 1), slot_count))}
        if program_parameters["matrix_modulation"]:
            effective.add(("M", 0))
    else:
        output_type = {1: "S", 2: "V", 3: "M"}.get(program_parameters["continuous_output"])
        if output_type:
            effective.add((output_type, 1 % slot_count))
    if program_parameters["self_modifying"]:
        effective.update(("S", index) for index in range(1, min(5, slot_count)))

    stateless_effective: set[int] = set()
    for index in range(len(decoded) - 1, -1, -1):
        output, inputs = decoded[index]
        if output in effective:
            stateless_effective.add(index)
            for memory_type, memory_index in inputs:
                if memory_type != "M" or memory_index == 0:
                    effective.add((memory_type, memory_index))

    selected: list[int] = []
    # Match MarkIntrons' forward fixed-point pass for stateful programs.
    for _ in instructions:
        selected = []
        for index, (output, inputs) in enumerate(decoded):
            if output[0] == "?" or output in effective or index in stateless_effective:
                selected.append(index)
                effective.update(inputs)
    return selected


def best_row(selection_file: Path) -> dict[str, str]:
    """Return the earliest generation among the rows with max best_fitness."""
    with selection_file.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("selection log is empty")
    return min(rows, key=lambda row: (-float(row["best_fitness"]), int(row["generation"])))


def newest_training_checkpoint(checkpoint_dir: Path, seed: int) -> Path:
    """Use the most recent phase-0 checkpoint, matching the replay command."""
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob(f"cp.*.{seed}.0.rslt"):
        match = CHECKPOINT_RE.match(path.name)
        if match:
            candidates.append((int(match["generation"]), path))
    if not candidates:
        raise FileNotFoundError(f"no training checkpoint found for seed {seed}")
    return max(candidates)[1]


def parse_checkpoint(
    checkpoint_file: Path, self_modifying: bool = False,
) -> tuple[dict[int, list[int]], dict[int, dict[str, object]]]:
    """Parse team memberships and RegisterMachine records from a checkpoint."""
    teams: dict[int, list[int]] = {}
    programs: dict[int, dict[str, object]] = {}
    memory_sizes: dict[int, str] = {}
    memory_constants: dict[int, dict[int, list[str]]] = {}

    for line in checkpoint_file.read_text(encoding="utf-8").splitlines():
        fields = line.split(":")
        if fields[0] == "team" and len(fields) >= 8:
            # team:id:generation:obs_index:lambda:decay:n_eval:program_id...
            teams[int(fields[1])] = [int(program_id) for program_id in fields[7:] if program_id]
        elif fields[0] == "MemoryEigen" and len(fields) >= 5:
            # MemoryEigen:program_id:type:memory_slots:memory_size:...
            # All memory types for a program share the same memory size.
            program_id = int(fields[1])
            memory_sizes.setdefault(program_id, fields[4])
            memory_type = int(fields[2])
            memory_constants.setdefault(program_id, {}).setdefault(
                memory_type, parse_memory_constants(fields, self_modifying)
            )
        elif fields[0] == "RegisterMachine" and len(fields) >= 12:
            # See RegisterMachine::ToString for the serialization layout.
            programs[int(fields[1])] = {
                "generation": fields[2],
                "action": int(fields[3]),
                "stateful": fields[4],
                "observation_buffer": fields[6],
                "learning_rate": fields[7],
                "observation_index": fields[9],
                "memory_slots": fields[10],
                "stateful_flags": fields[11],
                "instructions": fields[12:],
            }
    for program_id, program in programs.items():
        program["memory_size"] = memory_sizes.get(program_id, "unknown")
        constants_by_type = memory_constants.get(program_id, {})
        program["constants"] = [
            constant
            for memory_type in sorted(constants_by_type)
            for constant in constants_by_type[memory_type]
        ]
    return teams, programs


def reachable_programs(root_team: int, teams: dict[int, list[int]], programs: dict[int, dict[str, object]]):
    pending = deque([root_team])
    seen_teams: set[int] = set()
    seen_programs: set[int] = set()

    while pending:
        team_id = pending.popleft()
        if team_id in seen_teams:
            continue
        seen_teams.add(team_id)
        for program_id in teams.get(team_id, []):
            if program_id in seen_programs:
                continue
            seen_programs.add(program_id)
            program = programs.get(program_id)
            yield team_id, program_id, program
            if program and int(program["action"]) >= 0:
                pending.append(int(program["action"]))


def write_seed_report(output, selection_file: Path, experiment_dir: Path,
                      definitions: dict[int, tuple[str, list[str]]],
                      observation_dimension: int,
                      program_parameters: dict[str, int]) -> None:
    row = best_row(selection_file)
    seed = int(row["seed"]) if "seed" in row else int(selection_file.name.split(".")[1])
    team_id = int(float(row["team_id"]))
    checkpoint = newest_training_checkpoint(experiment_dir / "checkpoints", seed)
    teams, programs = parse_checkpoint(
        checkpoint, program_parameters["self_modifying"]
    )

    output.write(f"{'=' * 88}\n")
    output.write(f"seed: {seed}\n")
    output.write(f"best fitness: {row['best_fitness']}\n")
    output.write(f"best generation: {row['generation']}\n")
    output.write(f"best root team: {team_id}\n")
    output.write(f"selection log: {selection_file}\n")
    output.write(f"checkpoint used: {checkpoint}\n\n")

    if team_id not in teams:
        output.write("ERROR: this team is not present in the final checkpoint.\n\n")
        return

    for owner_team, program_id, program in reachable_programs(team_id, teams, programs):
        if program is None:
            output.write(f"Team {owner_team} -> missing program {program_id}\n\n")
            continue
        action = int(program["action"])
        action_label = f"subteam {action}" if action >= 0 else f"atomic action {action}"
        output.write(f"Team {owner_team} | Program {program_id} -> {action_label}\n")
        output.write(
            "metadata: "
            f"born={program['generation']}, stateful={program['stateful']}, "
            f"obs_index={program['observation_index']}, "
            f"obs_buffer={program['observation_buffer']}, "
            f"memory_slots={program['memory_slots']}, "
            f"memory_size={program['memory_size']}, "
            f"learning_rate={program['learning_rate']}\n"
        )
        constants = program["constants"]
        output.write(
            "constants: "
            f"{', '.join(constants) if constants else '<none found>'}\n"
        )
        output.write("instructions:\n")
        instructions = program["instructions"]
        if instructions:
            for index, instruction in enumerate(instructions):
                readable = render_instruction(
                    instruction, int(program["memory_slots"]), observation_dimension, definitions
                )
                output.write(f"  {index:>3}: {readable}\n")
        else:
            output.write("  <none>\n")
        output.write("instructions (introns removed):\n")
        effective_indices = effective_instruction_indices(
            instructions, int(program["memory_slots"]), definitions, program_parameters
        )
        if effective_indices:
            for index in effective_indices:
                readable = render_instruction(
                    instructions[index], int(program["memory_slots"]), observation_dimension, definitions
                )
                output.write(f"  {index:>3}: {readable}\n")
        else:
            output.write("  <none>\n")
        output.write("\n")


def main() -> None:
    experiment_dir = Path(EXPERIMENT_DIR)
    selection_dir = experiment_dir / "logs" / "selection"
    selection_files = sorted(selection_dir.glob("selection.*.*.csv"))
    if not selection_files:
        raise SystemExit(f"No selection logs found in {selection_dir}")

    output_path = experiment_dir / OUTPUT_FILENAME
    definitions = load_instruction_definitions()
    program_parameters = load_program_parameters(experiment_dir)
    with output_path.open("w", encoding="utf-8") as output:
        output.write("Best agent instructions by seed\n")
        output.write(f"Experiment directory: {experiment_dir}\n")
        output.write(f"Observation dimension: {OBSERVATION_DIMENSION}\n\n")
        for selection_file in selection_files:
            try:
                write_seed_report(
                    output, selection_file, experiment_dir, definitions, OBSERVATION_DIMENSION,
                    program_parameters,
                )
            except (ValueError, FileNotFoundError, KeyError) as error:
                output.write(f"{'=' * 88}\n{selection_file.name}: ERROR: {error}\n\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
