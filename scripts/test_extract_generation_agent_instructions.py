"""Regression tests for overwrite-aware instruction liveness."""
import unittest

from extract_generation_agent_instructions import analyze, load_instruction_definitions


class LivenessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.definitions = load_instruction_definitions()
        cls.opcodes = {name: opcode for opcode, (name, _) in cls.definitions.items()}

    def instruction(self, name, out=1, source=1, index=0):
        return f'{source}_1_{out}_{self.opcodes[name]}_{index}_0_0_0_'

    def retained(self, instructions, stateful=False, additive=False):
        return analyze(instructions, 10, self.definitions, {('S', 1)}, stateful, False, additive)

    def test_overwrite_discards_previous_value(self):
        instructions = [self.instruction('SCALAR_SIN_OP'), self.instruction('SCALAR_COS_OP')]
        self.assertEqual(self.retained(instructions), [1])

    def test_accumulation_preserves_previous_value(self):
        instructions = [self.instruction('SCALAR_SIN_OP'), self.instruction('SCALAR_ACCUM_OP')]
        self.assertEqual(self.retained(instructions), [0, 1])
        instructions.append(self.instruction('SCALAR_COS_OP'))
        self.assertEqual(self.retained(instructions), [2])

    def test_conditional_preserves_previous_destination(self):
        instructions = [self.instruction('SCALAR_SIN_OP'), self.instruction('SCALAR_CONDITIONAL_OP')]
        self.assertEqual(self.retained(instructions), [0, 1])

    def test_state_can_be_consumed_on_next_execution(self):
        instructions = [self.instruction('SCALAR_SIN_OP', source=0, index=7),
                        self.instruction('SCALAR_COS_OP', out=7)]
        self.assertEqual(self.retained(instructions), [0])
        self.assertEqual(self.retained(instructions, stateful=True), [0, 1])

    def test_constant_set_does_not_read_working_memory(self):
        instructions = [self.instruction('SCALAR_SIN_OP', out=7),
                        self.instruction('SCALAR_CONST_SET_OP', source=0, index=7)]
        self.assertEqual(self.retained(instructions), [1])

    def test_additive_mode_preserves_previous_value(self):
        instructions = [self.instruction('SCALAR_SIN_OP'), self.instruction('SCALAR_COS_OP')]
        self.assertEqual(self.retained(instructions, additive=True), [0, 1])

    def test_random_draw_is_preserved_when_destination_is_dead(self):
        instructions = [self.instruction('SCALAR_UNIFORM_SET_OP', out=7),
                        self.instruction('SCALAR_COS_OP')]
        self.assertEqual(self.retained(instructions), [0, 1])


if __name__ == '__main__':
    unittest.main()
