import argparse
import contextlib
import importlib.util
import io
from pathlib import Path
import re
import tempfile
import sys
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run/transfer_mountaincar_to_acrobot.py'
spec = importlib.util.spec_from_file_location('transfer', SCRIPT)
transfer = importlib.util.module_from_spec(spec)
with patch.object(sys, 'path', [str(SCRIPT.parent), *sys.path]):
    spec.loader.exec_module(transfer)


class MountainCarAcrobotTransferRunnerTest(unittest.TestCase):
    def test_default_plans_twenty_distinct_runs(self):
        for variant in transfer.SOURCE_CONFIGS:
            with self.subTest(variant=variant), patch('sys.argv', [str(SCRIPT), variant, '--dry-run']), \
                 patch.object(transfer, 'run_transfer') as run, \
                 patch.object(transfer, 'source_checkpoint', return_value=Path('/test/checkpoint')), \
                 patch.object(transfer, 'match_checkpoint_registers', side_effect=lambda config, path: config), \
                 contextlib.redirect_stdout(io.StringIO()):
                transfer.main()
                self.assertEqual(sorted(call.args[1] for call in run.call_args_list), list(range(1, 21)))
                self.assertEqual({call.args[0] for call in run.call_args_list}, {variant})
                self.assertTrue(all(call.args[2].generations == 10000 for call in run.call_args_list))

    def test_configs_keep_mutation_settings_and_disable_non_scalar_ops(self):
        for variant in transfer.SOURCE_CONFIGS:
            mountaincar, acrobot = transfer.experiment_configs(variant)
            self.assertIn('active_tasks: "MountainCarContinuous"', mountaincar)
            self.assertIn('active_tasks: "Acrobot"', acrobot)
            self.assertIn('n_input: "4"', acrobot)
            self.assertIn('n_generations: 5000', mountaincar)
            self.assertIn('n_generations: 10000', acrobot)
            self.assertIn('acrobot_n_eval_validation: 0', acrobot)
            self.assertIn('mountaincar_continuous_max_timesteps: 200', mountaincar)
            self.assertIn('acrobot_max_timesteps: 500', acrobot)
            for content in (mountaincar, acrobot):
                self.assertIn('validation_mod: 0', content)
                self.assertIn('test_mod: 0', content)
                budgets = re.findall(r'^  \w+_n_eval_(?:validation|val|test): (\d+)', content, re.MULTILINE)
                self.assertTrue(budgets)
                self.assertEqual(set(budgets), {'0'})
                self.assertRegex(content, r'_n_eval_train: 20\b')
                enabled = re.findall(r'^  (\w+_OP): 1\b', content, re.MULTILINE)
                self.assertTrue(enabled)
                self.assertTrue(all(name.startswith('SCALAR_') and 'VECTOR' not in name and 'MATRIX' not in name
                                    and name != 'SCALAR_BROADCAST_OP' for name in enabled))
            # Mutation and state settings remain identical across the two stages.
            before = mountaincar.split('# General Task Parameters')[0]
            after = acrobot.split('# General Task Parameters')[0]
            before = re.sub(r'^  n_generations:.*$', '', before, flags=re.MULTILINE)
            after = re.sub(r'^  n_generations:.*$', '', after, flags=re.MULTILINE)
            self.assertEqual(before.replace(
                'Scalar S1 controls force in [-1, 1]',
                'Scalar S1 controls torque, clamped by the task to [-1, 1]').replace(
                '  n_input: "4"  # Observation count in this repository\'s environment',
                '  n_input: "4"  # Acrobot has four observations'), after)

    def test_stages_transfer_full_checkpoint_and_resume_without_repeating(self):
        for variant in transfer.SOURCE_CONFIGS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                args = argparse.Namespace(output_dir=Path(temporary), executable=Path('/test/binary'),
                                          generations=10000, seed_aux=42, processes=2, dry_run=False,
                                          fresh=True, launcher='mpirun')
                calls = []

                def simulate(command, *, cwd, **kwargs):
                    settings = dict(value.split('=', 1) for value in command if '=' in value)
                    generation = int(settings['n_generations'])
                    config = Path(settings['parameters_file']).read_text()
                    calls.append((cwd.name, settings, config))
                    if cwd.name == 'acrobot':
                        self.assertEqual(settings['checkpoint_in_t'], '5000')
                        self.assertEqual(settings['start_from_checkpoint'], '1')
                        self.assertEqual((cwd / 'checkpoints/cp.5000.7.0.rslt').read_text(),
                                         'seed_tpg:7\nt:5000\nfull-population\nend\n')
                    (cwd / f'checkpoints/cp.{generation}.7.0.rslt').write_text(
                        f'seed_tpg:7\nt:{generation}\nfull-population\nend\n')

                with patch.object(transfer.subprocess, 'run', side_effect=simulate), \
                     contextlib.redirect_stdout(io.StringIO()):
                    transfer.run_transfer(variant, 7, args)
                    transfer.run_transfer(variant, 7, args)
                self.assertEqual([call[0] for call in calls], ['mountaincar_continuous', 'acrobot'])
                self.assertEqual(calls[0][1]['n_generations'], '5000')
                self.assertEqual(calls[0][1]['start_from_checkpoint'], '0')
                self.assertEqual(calls[1][1]['n_generations'], '10000')
                self.assertEqual(calls[0][1]['seed_tpg'], calls[1][1]['seed_tpg'])
                self.assertEqual(calls[0][1]['seed_aux'], calls[1][1]['seed_aux'])

    def test_existing_checkpoint_transfers_directly_and_preserves_vector_instructions(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source_dir = directory / 'sources'
            source = source_dir / 'mountaincar_continuous_fixed_rates/checkpoints/cp.5000.7.0.rslt'
            source.parent.mkdir(parents=True)
            saved = ('seed_tpg:7\nseed_aux:42\nt:5000\nMemoryEigen:123:0:11:1:0\n'
                     'RegisterMachine:123:0:-1:1:1:1:0:0:0:11:SM0:SL4:0_0_5_22_1_5_2_8_\nend\n')
            source.write_text(saved)
            args = argparse.Namespace(output_dir=directory / 'outputs', source_dir=source_dir,
                                      executable=Path('/test/binary'), generations=10000,
                                      seed_aux=42, processes=21, dry_run=False, fresh=False, launcher='srun')
            calls = []

            def simulate(command, *, cwd, **kwargs):
                self.assertEqual(command[:4], ['srun', '--ntasks', '21', '--kill-on-bad-exit=1'])
                self.assertNotIn('mpirun', command)
                self.assertNotIn('--oversubscribe', command)
                settings = dict(value.split('=', 1) for value in command if '=' in value)
                calls.append(settings)
                self.assertEqual(cwd.name, 'acrobot')
                self.assertEqual(settings['checkpoint_in_t'], '5000')
                self.assertEqual((cwd / 'checkpoints/cp.5000.7.0.rslt').read_text(), saved)
                config = Path(settings['parameters_file']).read_text()
                for key in ('min_initial_mem_slots', 'max_initial_mem_slots', 'min_memory_slots', 'max_memory_slots'):
                    self.assertIn(f'{key}: 11', config)
                (cwd / 'checkpoints/cp.10000.7.0.rslt').write_text('seed_tpg:7\nt:10000\nend\n')

            with patch.object(transfer.subprocess, 'run', side_effect=simulate), \
                 contextlib.redirect_stdout(io.StringIO()):
                transfer.run_transfer('fixed_rates', 7, args)
                transfer.run_transfer('fixed_rates', 7, args)
            self.assertEqual(len(calls), 1)
            self.assertEqual(source.read_text(), saved)

    def test_source_requires_generation_5000(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / 'mountaincar_continuous_fixed_rates/checkpoints/cp.5000.1.0.rslt'
            source.parent.mkdir(parents=True)
            source.write_text('seed_tpg:1\nt:5000\nend\n')
            self.assertEqual(transfer.source_checkpoint('fixed_rates', 1, directory), source)
            source.write_text('seed_tpg:1\nt:5001\nend\n')
            with self.assertRaisesRegex(ValueError, 'requires generation 5000'):
                transfer.source_checkpoint('fixed_rates', 1, directory)

    def test_batch_preflight_rejects_missing_source_before_any_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / 'mountaincar_continuous_fixed_rates/checkpoints/cp.5000.1.0.rslt'
            source.parent.mkdir(parents=True)
            source.write_text('seed_tpg:1\nt:5000\nMemoryEigen:123:0:10:1:0\nend\n')
            command = [str(SCRIPT), 'fixed_rates', '--seeds', '1', '2',
                       '--source-dir', str(directory), '--executable', str(SCRIPT)]
            with patch('sys.argv', command), patch.object(transfer.shutil, 'which', return_value='/test/mpirun'), \
                 patch.object(transfer, 'run_transfer') as run, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    transfer.main()
                self.assertEqual(error.exception.code, 1)
                run.assert_not_called()

    def test_incomplete_or_wrong_seed_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'cp.5000.1.0.rslt'
            checkpoint.write_text('seed_tpg:1\nt:5000\npartial\n')
            self.assertFalse(transfer.completed_checkpoint(checkpoint))
            with self.assertRaises(ValueError):
                transfer.checkpoint_generation(checkpoint, 1)
            checkpoint.write_text('seed_tpg:2\nt:5000\nend\n')
            with self.assertRaises(ValueError):
                transfer.checkpoint_generation(checkpoint, 1)


if __name__ == '__main__':
    unittest.main()
