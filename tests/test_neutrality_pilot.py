"""Integration checks against real saved checkpoints; no training is run."""
import csv
import json
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/analysis/run_neutrality_pilot.py"


def read_csv(path):
    with path.open() as file:
        return list(csv.DictReader(file))


class NeutralityPilotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="lgp-neutrality-test-")
        cls.first = Path(cls.temp.name) / "first"
        cls.second = Path(cls.temp.name) / "second"
        for out in (cls.first, cls.second):
            subprocess.run([sys.executable, str(RUNNER), "smoke", "--output", str(out)],
                           cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_reproducibility_and_resume(self):
        for file in self.first.rglob("*.csv"):
            other = self.second / file.relative_to(self.first)
            self.assertEqual(file.read_bytes(), other.read_bytes())
        complete = list(self.first.rglob("complete.json"))
        before = [p.stat().st_mtime_ns for p in complete]
        subprocess.run([sys.executable, str(RUNNER), "smoke", "--output", str(self.first)],
                       cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(before, [p.stat().st_mtime_ns for p in complete])

    def test_pairing_structure_and_genotype_identity(self):
        for file in self.first.rglob("individuals.csv"):
            rows = read_csv(file)
            self.assertEqual(len(rows), 3)
            genomes = [json.loads(line) for line in (file.parent / "genomes.jsonl").read_text().splitlines()]
            parent = next(r["genotype"] for r in genomes if r["offspring"] == -1)
            for row, genome in zip(rows, genomes):
                self.assertEqual(int(row["total"]), sum(int(row[k]) for k in ("action_effective", "mutation_only", "introns")))
                self.assertEqual(int(row["genotype_changed"]), int(genome["genotype"] != parent))
            episodes = read_csv(file.parent / "episodes.csv")
            baseline = {(r["environment"], r["episode"]): r for r in episodes if r["offspring"] == "-1"}
            expected_envs = 2 if file.parent.name == "generation_2000" else 1
            self.assertEqual(len(episodes), 6 * expected_envs)
            for row in episodes:
                ref = baseline[row["environment"], row["episode"]]
                self.assertEqual(row["episode_seed"], ref["episode_seed"])
                self.assertEqual(float(row["parent_return"]), float(ref["return"]))
                self.assertAlmostEqual(float(row["delta_return"]), float(row["return"]) - float(ref["return"]))
                if rows[int(row["offspring"]) + 1]["genotype_changed"] == "0":
                    self.assertEqual(float(row["delta_return"]), 0)
                    self.assertEqual(float(row["probe_action_mae"]), 0)

    def test_twenty_unique_random_parents(self):
        out = Path(self.temp.name) / "twenty"
        command = [sys.executable, str(RUNNER), "run", "--tasks", "pendulum", "--seeds", "1",
                   "--generations", "2100", "--parents", "20", "--offspring", "2", "--episodes", "1", "--output", str(out)]
        subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        rows = read_csv(next(out.rglob("individuals.csv")))
        parents = [r for r in rows if r["offspring"] == "-1"]
        self.assertEqual(len(parents), 20)
        self.assertEqual(len({r["team_id"] for r in parents}), 20)
        self.assertEqual(len(rows), 60)
        # Resume must refuse a changed experiment specification.
        command[command.index("--offspring") + 1] = "3"
        failed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("Inputs/settings/code changed", failed.stderr)

    def test_zero_mutation_control(self):
        directory = Path(self.temp.name) / "zero_mutation"
        directory.mkdir()
        base = ROOT / "experiments/mountaincar_continuous_to_pendulum_execution_modified_rates/seed_1/pendulum"
        config = (base / "parameters.yaml").read_text()
        for key in ("p_instructions_swap", "p_instructions_delete", "p_instructions_add", "p_instructions_mutate",
                    "p_memory_mu_const", "mut_learning_rate", "p_memory_size", "p_memory_slots",
                    "p_hidden_stateful_mutation", "p_observation_index"):
            config, count = re.subn(rf"(?m)^(\s*{key}:).*", rf"\g<1> 0.0", config)
            self.assertEqual(count, 1)
        config = re.sub(r'(?m)^(\s*self_modifying_mutation_rates:).*', r'\1 ""', config)
        path = directory / "parameters.yaml"; path.write_text(config)
        subprocess.run([str(ROOT / "build/neutrality-pilot/neutrality_worker"), str(path),
                        str(base / "checkpoints/cp.2100.1.0.rslt"), "-", "1", "2100", "2", "2", "2", "123", "456", "789"],
                       cwd=directory, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for row in read_csv(directory / "individuals.csv"):
            self.assertEqual(row["genotype_changed"], "0")
            for rate in ("swap", "delete", "add", "mutate"):
                self.assertEqual(float(row[f"used_{rate}"]), 0)
        for row in read_csv(directory / "episodes.csv"):
            for field in ("delta_return", "probe_action_mae", "probe_swap_mae", "probe_delete_mae", "probe_add_mae", "probe_mutate_mae"):
                self.assertEqual(float(row[field]), 0)


if __name__ == "__main__":
    unittest.main()
