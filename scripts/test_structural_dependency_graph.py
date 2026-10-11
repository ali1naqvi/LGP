"""Component mappings checked against small hand-derived engine dataflows."""
import unittest

from analyze_observation_rate_semantics import build_graph, metrics, load_instruction_definitions
from structural_dependency_graph import build_component_graph, engine_effective_indices


class ComponentGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.definitions = load_instruction_definitions()
        cls.codes = {name: op for op, (name, _) in cls.definitions.items()}

    def ins(self, name, out=2, a=1, x=0, b=1, y=1, z=0, w=0):
        return f'{a}_{b}_{out}_{self.codes[name]}_{x}_{y}_{z}_{w}_'

    def result(self, instructions, output=2, **kwargs):
        nodes, outputs = build_component_graph(instructions, 10, self.definitions, 3, memory_size=2, **kwargs)
        return metrics(nodes, outputs[output], 3)

    def test_scalar_analysis_matches_previous_graph(self):
        ins = [self.ins('SCALAR_SIN_OP'), self.ins('SCALAR_ACCUM_OP', a=0, x=2)]
        old_nodes, old_out = build_graph(ins, 10, self.definitions, 3)
        self.assertEqual(self.result(ins), metrics(old_nodes, old_out[2], 3))

    def test_matrix_observation_reduction_wraps_channels(self):
        rows, initial = self.result([self.ins('MATRIX_MEAN_OP')])
        self.assertEqual([(r['size'], r['depth'], r['paths']) for r in rows],
                         [(1, 1, 2), (1, 1, 1), (1, 1, 1), (1, 1, 4)])
        self.assertEqual(initial, [])

    def test_component_extract_uses_effective_zero_index(self):
        rows, _ = self.result([self.ins('SCALAR_VECTOR_ASSIGN_OP', z=1, w=1)])
        self.assertEqual([r['paths'] for r in rows], [1, 0, 0, 1])

    def test_vector_reciprocal_preserves_other_components(self):
        rows, initial = self.result([self.ins('VECTOR_RECIPROCAL_OP', out=7),
                                    self.ins('VECTOR_MEAN_OP', a=0, x=7)])
        self.assertEqual(rows[-1]['instruction_indices'], [0, 1])
        self.assertEqual(rows[-1]['paths'], 1)
        self.assertEqual(initial, ['V7'])

    def test_in_place_transpose_reads_previous_destination(self):
        rows, initial = self.result([self.ins('MATRIX_TRANSPOSE_OP', out=0),
                                    self.ins('MATRIX_MEAN_OP', a=0, x=0)])
        self.assertEqual(rows[-1]['paths'], 0)
        self.assertEqual(initial, ['M0'])

    def test_mixed_operation_uses_unpopulated_scalar_observation_buffer(self):
        rows, _ = self.result([self.ins('SCALAR_BROADCAST_OP', out=7),
                              self.ins('VECTOR_MEAN_OP', a=0, x=7)])
        self.assertEqual(rows[-1]['paths'], 0)

    def test_engine_intron_mask_is_applied(self):
        ins = [self.ins('SCALAR_SIN_OP'), self.ins('SCALAR_COS_OP')]
        rows, _ = self.result(ins, selected=[0])
        self.assertEqual(rows[-1]['instruction_indices'], [0])

    def test_engine_marking_excludes_decoy_writes(self):
        ins = [self.ins('SCALAR_SIN_OP', out=6), self.ins('SCALAR_SIN_OP')]
        selected = engine_effective_indices(ins, 10, self.definitions, {('S', 2)}, True)
        self.assertEqual(selected, [1])


if __name__ == '__main__':
    unittest.main()
