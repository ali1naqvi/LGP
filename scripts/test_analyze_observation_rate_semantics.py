"""Known dependency graphs validate size, depth, paths, and overwrite rules."""
import unittest

from analyze_observation_rate_semantics import build_graph, metrics, register_metrics, load_instruction_definitions


class SemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.definitions = load_instruction_definitions()
        cls.opcodes = {name: op for op, (name, _) in cls.definitions.items()}

    def instruction(self, name, dst=2, src0=1, idx0=0, src1=1, idx1=1):
        return f'{src0}_{src1}_{dst}_{self.opcodes[name]}_{idx0}_{idx1}_0_0_'

    def evaluate(self, instructions, output=2, **kwargs):
        nodes, versions = build_graph(instructions, 10, self.definitions, 3, **kwargs)
        rows, initial = metrics(nodes, versions[output], 3)
        return {row['observation']: row for row in rows}, initial

    def counts(self, row):
        return row['size'], row['depth'], row['paths']

    def test_single_observation(self):
        rows, initial = self.evaluate([self.instruction('SCALAR_SIN_OP')])
        self.assertEqual(self.counts(rows['0']), (1, 1, 1))
        self.assertEqual(self.counts(rows['1']), (0, 0, 0))
        self.assertEqual(initial, [])

    def test_overwrite_severs_old_dependency(self):
        rows, _ = self.evaluate([self.instruction('SCALAR_SIN_OP'),
                                self.instruction('SCALAR_COS_OP', src0=2)])
        self.assertEqual(self.counts(rows['all']), (0, 0, 0))

    def test_accumulation_duplicates_paths_and_preserves_value(self):
        rows, _ = self.evaluate([self.instruction('SCALAR_SIN_OP'),
            self.instruction('SCALAR_ACCUM_OP', src0=0, idx0=2)])
        self.assertEqual(self.counts(rows['0']), (2, 2, 2))

    def test_two_observations_share_one_instruction(self):
        rows, _ = self.evaluate([self.instruction('SCALAR_SUM_OP')])
        self.assertEqual(self.counts(rows['all']), (1, 1, 2))
        self.assertEqual(self.counts(rows['0']), (1, 1, 1))
        self.assertEqual(self.counts(rows['1']), (1, 1, 1))

    def test_diamond_counts_unique_nodes_and_both_paths(self):
        rows, _ = self.evaluate([
            self.instruction('SCALAR_SIN_OP', dst=7),
            self.instruction('SCALAR_COS_OP', dst=8, src0=0, idx0=7),
            self.instruction('SCALAR_SUM_OP', src0=0, idx0=7, src1=0, idx1=8)])
        self.assertEqual(self.counts(rows['0']), (3, 3, 2))

    def test_prior_state_is_separate(self):
        rows, initial = self.evaluate([self.instruction('SCALAR_ACCUM_OP')])
        self.assertEqual(self.counts(rows['0']), (1, 1, 1))
        self.assertEqual(initial, ['S2'])

    def test_conditional_reads_condition_and_previous_destination(self):
        rows, _ = self.evaluate([self.instruction('SCALAR_SIN_OP'),
            self.instruction('SCALAR_CONDITIONAL_OP', idx0=1, idx1=2)])
        self.assertEqual(self.counts(rows['0']), (2, 2, 1))
        self.assertEqual(self.counts(rows['all']), (2, 2, 3))

    def test_runtime_decoy_and_hypothetical_writes(self):
        ins = [self.instruction('SCALAR_SIN_OP', dst=6)]
        rows, initial = self.evaluate(ins, output=6)
        self.assertEqual(self.counts(rows['all']), (0, 0, 0))
        self.assertEqual(initial, ['S6'])
        rows, initial = self.evaluate(ins, output=6, hypothetical_decoy=True)
        self.assertEqual(self.counts(rows['all']), (1, 1, 1))
        self.assertEqual(initial, [])

    def test_hypothetical_decoy_reads_do_not_change_rates(self):
        ins = [self.instruction('SCALAR_SIN_OP', dst=6),
               self.instruction('SCALAR_SIN_OP', src0=0, idx0=6)]
        rows, _ = self.evaluate(ins, hypothetical_decoy=True)
        self.assertEqual(self.counts(rows['all']), (0, 0, 0))

    def test_additive_operations_include_old_destination(self):
        ins = [self.instruction('SCALAR_SIN_OP'),
               self.instruction('SCALAR_COS_OP', idx0=1)]
        rows, _ = self.evaluate(ins, additive=True)
        self.assertEqual(self.counts(rows['0']), (2, 2, 1))
        self.assertEqual(self.counts(rows['all']), (2, 2, 2))

    def test_register_versions_respect_overwrites(self):
        ins = [self.instruction('SCALAR_SIN_OP', dst=7),
               self.instruction('SCALAR_COS_OP', dst=7),
               self.instruction('SCALAR_SIN_OP', src0=0, idx0=7)]
        nodes, outputs = build_graph(ins, 10, self.definitions, 3)
        rows = register_metrics(nodes, {2: outputs[2]}, 10)
        row = next(r for r in rows if r['source_scope'] == 'written_values' and r['source_register'] == 'S7')
        self.assertEqual(self.counts(row), (1, 1, 1))
        self.assertEqual(row['source_nodes'], ['instruction:1'])

    def test_register_accumulation_incoming_and_written_sources(self):
        ins = [self.instruction('SCALAR_SIN_OP', dst=7),
               self.instruction('SCALAR_ACCUM_OP', src0=0, idx0=7)]
        nodes, outputs = build_graph(ins, 10, self.definitions, 3)
        rows = register_metrics(nodes, {2: outputs[2]}, 10)
        incoming = next(r for r in rows if r['source_scope'] == 'incoming_value' and r['source_register'] == 'S2')
        written = next(r for r in rows if r['source_scope'] == 'written_values' and r['source_register'] == 'S7')
        self.assertEqual(self.counts(incoming), (1, 1, 1))
        self.assertEqual(self.counts(written), (1, 1, 1))

    def test_register_unchanged_identity(self):
        nodes, outputs = build_graph([], 10, self.definitions, 3)
        rows = register_metrics(nodes, {6: outputs[6]}, 10)
        row = next(r for r in rows if r['source_scope'] == 'incoming_value' and r['source_register'] == 'S6')
        self.assertEqual(self.counts(row), (0, 0, 1))
        self.assertEqual(row['identity_paths'], 1)


if __name__ == '__main__':
    unittest.main()
