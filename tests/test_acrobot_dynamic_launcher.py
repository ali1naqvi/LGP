import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DynamicLauncherTest(unittest.TestCase):
    def test_array_mapping_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'configs').mkdir()
            binary = root / 'build/release/experiments/TPGExperimentMPI'
            binary.parent.mkdir(parents=True)
            binary.write_text('#!/bin/bash\nexit 0\n')
            binary.chmod(0o755)
            for variant in ('fixed_rates', 'fixed_individual', 'execution_modified_rates'):
                name = f'acrobot_dynamic_{variant}.yaml'
                (root / 'configs' / name).write_text((ROOT / 'configs' / name).read_text())
            mockbin = root / 'mockbin'
            mockbin.mkdir()
            for name, body in {'git': 'echo test-revision', 'srun': 'printf "%s\\n" "$@"'}.items():
                script = mockbin / name
                script.write_text('#!/bin/bash\n' + body + '\n')
                script.chmod(0o755)
            env = dict(os.environ, TPG=str(root), SLURM_NTASKS='21', SLURM_JOB_ID='123',
                       PATH=str(mockbin) + os.pathsep + os.environ['PATH'])
            for index, variant, seed in [(1, 'fixed_rates', 1), (20, 'fixed_rates', 20),
                                         (21, 'fixed_individual', 1), (40, 'fixed_individual', 20),
                                         (41, 'execution_modified_rates', 1),
                                         (60, 'execution_modified_rates', 20)]:
                env['SLURM_ARRAY_TASK_ID'] = str(index)
                command = ['bash', str(ROOT / 'scripts/run/acrobot-dynamic.slurm'), '200']
                subprocess.run(command, env=env, check=True, capture_output=True)
                run = root / 'experiments/acrobot_dynamic' / variant / f'seed_{seed}'
                log = next((run / 'logs').glob('*.std')).read_text()
                self.assertIn(f'seed_tpg={seed}\n', log)
                self.assertIn('n_generations=200\n', log)
                self.assertIn('--ntasks=21\n', log)
                self.assertNotEqual(subprocess.run(command, env=env, capture_output=True).returncode, 0)

    def test_ssh_dry_run(self):
        result = subprocess.run(['bash', str(ROOT / 'scripts/run/acrobot-dynamic-ssh.sh'),
                                 '--dry-run', 'user@cluster', '/remote/repo with spaces'],
                                check=True, capture_output=True, text=True)
        self.assertIn('ssh user@cluster', result.stdout)
        self.assertIn('sbatch', result.stdout)


if __name__ == '__main__':
    unittest.main()
