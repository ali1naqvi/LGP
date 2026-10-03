import argparse
import contextlib
import importlib.util
import io
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run/transfer_pendulum_to_acrobot.py'
spec = importlib.util.spec_from_file_location('transfer', SCRIPT)
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


class TransferRunnerTest(unittest.TestCase):
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

    def test_configs_keep_mutation_settings_and_disable_non_scalar_ops(self):
        for variant in transfer.SOURCE_CONFIGS:
            pendulum, acrobot = transfer.experiment_configs(variant)
            self.assertIn('active_tasks: "Pendulum"', pendulum)
            self.assertIn('active_tasks: "Acrobot"', acrobot)
            self.assertIn('n_input: "4"', acrobot)
            self.assertIn('acrobot_n_eval_validation: 5', acrobot)
            self.assertIn('pendulum_max_timesteps: 200', pendulum)
            self.assertIn('acrobot_max_timesteps: 500', acrobot)
            for content in (pendulum, acrobot):
                enabled = re.findall(r'^  (\w+_OP): 1\b', content, re.MULTILINE)
                self.assertTrue(enabled)
                self.assertTrue(all(name.startswith('SCALAR_') and 'VECTOR' not in name and 'MATRIX' not in name
                                    and name != 'SCALAR_BROADCAST_OP' for name in enabled))
            # Only the environment and observation count change across stages.
            self.assertEqual(pendulum.split('# General Task Parameters')[0].replace(
                'Scalar S1 controls torque, clamped by the task to [-2, 2]',
                'Scalar S1 controls torque, clamped by the task to [-1, 1]').replace(
                '  n_input: "3"  # cos(theta), sin(theta), angular velocity',
                '  n_input: "4"  # Acrobot has four observations'),
                acrobot.split('# General Task Parameters')[0])

    def test_stages_transfer_full_checkpoint_and_resume_without_repeating(self):
        for variant in transfer.SOURCE_CONFIGS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                args = argparse.Namespace(output_dir=Path(temporary), executable=Path('/test/binary'),
                                          generations=1002, seed_aux=42, processes=2, dry_run=False, fresh=True)
                calls = []

                def simulate(command, *, cwd, **kwargs):
                    settings = dict(value.split('=', 1) for value in command if '=' in value)
                    generation = int(settings['n_generations'])
                    config = Path(settings['parameters_file']).read_text()
                    calls.append((cwd.name, settings, config))
                    if cwd.name == 'acrobot':
                        self.assertEqual(settings['checkpoint_in_t'], '1000')
                        self.assertEqual(settings['start_from_checkpoint'], '1')
                        self.assertEqual((cwd / 'checkpoints/cp.1000.7.0.rslt').read_text(),
                                         'seed_tpg:7\nt:1000\nfull-population\nend\n')
                    (cwd / f'checkpoints/cp.{generation}.7.0.rslt').write_text(
                        f'seed_tpg:7\nt:{generation}\nfull-population\nend\n')

                with patch.object(transfer.subprocess, 'run', side_effect=simulate), \
                     contextlib.redirect_stdout(io.StringIO()):
                    transfer.run_transfer(variant, 7, args)
                    transfer.run_transfer(variant, 7, args)
                self.assertEqual([call[0] for call in calls], ['pendulum', 'acrobot'])
                self.assertEqual(calls[0][1]['n_generations'], '1000')
                self.assertEqual(calls[0][1]['start_from_checkpoint'], '0')
                self.assertEqual(calls[1][1]['n_generations'], '1002')
                self.assertEqual(calls[0][1]['seed_tpg'], calls[1][1]['seed_tpg'])
                self.assertEqual(calls[0][1]['seed_aux'], calls[1][1]['seed_aux'])

    def test_existing_checkpoint_transfers_directly_and_preserves_vector_instructions(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source_dir = directory / 'sources'
            source = source_dir / 'pendulum_fixed_rates_no_limit/checkpoints/cp.1000.7.0.rslt'
            source.parent.mkdir(parents=True)
            saved = ('seed_tpg:7\nseed_aux:42\nt:1000\nMemoryEigen:123:0:11:1:0\n'
                     'RegisterMachine:123:0:-1:1:1:1:0:0:0:11:SM0:SL4:0_0_5_22_1_5_2_8_\nend\n')
            source.write_text(saved)
            args = argparse.Namespace(output_dir=directory / 'outputs', source_dir=source_dir,
                                      executable=Path('/test/binary'), generations=1002,
                                      seed_aux=42, processes=2, dry_run=False, fresh=False)
            calls = []

            def simulate(command, *, cwd, **kwargs):
                settings = dict(value.split('=', 1) for value in command if '=' in value)
                calls.append(settings)
                self.assertEqual(cwd.name, 'acrobot')
                self.assertEqual(settings['checkpoint_in_t'], '1000')
                self.assertEqual((cwd / 'checkpoints/cp.1000.7.0.rslt').read_text(), saved)
                config = Path(settings['parameters_file']).read_text()
                for key in ('min_initial_mem_slots', 'max_initial_mem_slots', 'min_memory_slots', 'max_memory_slots'):
                    self.assertIn(f'{key}: 11', config)
                (cwd / 'checkpoints/cp.1002.7.0.rslt').write_text('seed_tpg:7\nt:1002\nend\n')

            with patch.object(transfer.subprocess, 'run', side_effect=simulate), \
                 contextlib.redirect_stdout(io.StringIO()):
                transfer.run_transfer('fixed_rates', 7, args)
                transfer.run_transfer('fixed_rates', 7, args)
            self.assertEqual(len(calls), 1)
            self.assertEqual(source.read_text(), saved)

    def test_incomplete_or_wrong_seed_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'cp.1000.1.0.rslt'
            checkpoint.write_text('seed_tpg:1\nt:1000\npartial\n')
            self.assertFalse(transfer.completed_checkpoint(checkpoint))
            with self.assertRaises(ValueError):
                transfer.checkpoint_generation(checkpoint, 1)
            checkpoint.write_text('seed_tpg:2\nt:1000\nend\n')
            with self.assertRaises(ValueError):
                transfer.checkpoint_generation(checkpoint, 1)


if __name__ == '__main__':
    unittest.main()
