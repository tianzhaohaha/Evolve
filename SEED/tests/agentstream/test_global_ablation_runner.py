"""CPU-only suite runner checks using a relocated checkout and fake launcher."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SEED_ROOT = Path(__file__).resolve().parents[2]
RUNNER_DIR = Path("examples/agentstream_trainer")


class GlobalAblationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "moved checkout" / "SEED"
        scripts = self.root / RUNNER_DIR
        scripts.mkdir(parents=True)
        for filename in ("run_global_ablation.sh", "agentstream_full.env"):
            shutil.copyfile(SEED_ROOT / RUNNER_DIR / filename, scripts / filename)
        self.script = scripts / "run_global_ablation.sh"
        self.capture = Path(self.temp.name) / "calls.jsonl"
        self.env = {
            "PATH": os.environ["PATH"], "HOME": self.temp.name,
            "CAPTURE": str(self.capture), "PBS_JOBID": "123.server",
            "EXPERIMENT_NAME": "must_not_leak", "DEFAULT_LOCAL_DIR": "/wrong",
        }
        # A machine file with hard assignments must not defeat run_exp overrides.
        (self.root / ".env").write_text(
            "export AGENTSTREAM_SEED_EMA_MODE=both\n"
            "export AGENTSTREAM_RL_EPOCHS=99\n"
            "export AGENTSTREAM_SEED_OPD_GEN_LOSS_COEF=99\n"
        )
        (scripts / "run_agentstream_sft_glm_self.sh").write_text(
            '#!/usr/bin/env bash\nset -eo pipefail\n'
            'source "$AGENTSTREAM_CONFIG"\n'
            'export TEST_MODE="$1"\n'
            "python3 - <<'PY'\n"
            "import json, os\n"
            "with open(os.environ['CAPTURE'], 'a') as f:\n"
            "    f.write(json.dumps(dict(os.environ)) + '\\n')\n"
            "PY\n"
            'echo "fake training"\n'
            'if [[ "$AGENTSTREAM_EXPERIMENT_PREFIX" == *"${FAIL_TAG:-never}" ]]; then\n'
            '    exit 7\nfi\n'
        )

    def run_suite(self, *args, **env):
        return subprocess.run(
            ["bash", str(self.script), *args], cwd=self.temp.name,
            env={**self.env, **env}, capture_output=True, text=True,
        )

    def test_four_runs_and_override_isolation(self):
        result = self.run_suite()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [json.loads(line) for line in self.capture.read_text().splitlines()]
        self.assertEqual(len(calls), 4)
        for row, coef, gen, source, eps, positive, ema, replay in zip(
            calls, ["0.01", "0.005", "0.005", "0.005"], ["0", "0.005", "0.005", "0.005"],
            ["copy", "pool", "pool", "pool"], ["0", "0", "0", "0.05"],
            ["False", "False", "False", "True"], ["off", "off", "off", "ref"],
            ["False", "False", "True", "True"],
        ):
            for key, value in {
                "OPD_LOSS_COEF": coef, "OPD_GEN_LOSS_COEF": gen,
                "GLOBAL_POOL_SOURCE": source, "OPD_GATE_EPS": eps,
                "OPD_POSITIVE_ONLY": positive, "EMA_MODE": ema, "REPLAY_ENABLE": replay,
                "REPLAY_GROUPS_PER_STEP": "2",
            }.items():
                self.assertEqual(row["AGENTSTREAM_SEED_" + key], value)
            self.assertEqual(row["AGENTSTREAM_RL_EPOCHS"], "20")
            self.assertEqual(row["AGENTSTREAM_RL_TRAIN_DATA_SIZE"], "10")
            self.assertEqual(row["TEST_MODE"], "interleaved")
            self.assertEqual(row["ENV_FILE"], "/dev/null")
            self.assertNotIn("EXPERIMENT_NAME", row)
            self.assertNotIn("DEFAULT_LOCAL_DIR", row)
            self.assertIn("_b10_steps20_", row["AGENTSTREAM_EXPERIMENT_PREFIX"])
        self.assertEqual(len({row["AGENTSTREAM_EXPERIMENT_PREFIX"] for row in calls}), 4)
        self.assertEqual(len(list((self.root / "logs/agentstream").glob("*123.server.log"))), 4)

    def test_training_failure_continues_and_returns_failure(self):
        result = self.run_suite(FAIL_TAG="global_clean")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(self.capture.read_text().splitlines()), 4)
        self.assertIn("training=7 tee=0", result.stdout)
        self.assertIn("[SUCCESS] SEED+Global+Opt", result.stdout)

    def test_tee_failure_is_reported(self):
        bin_dir = Path(self.temp.name) / "bin"
        bin_dir.mkdir()
        tee = bin_dir / "tee"
        tee.write_text("#!/bin/bash\ncat\nexit 9\n")
        tee.chmod(0o755)
        result = self.run_suite(PATH=str(bin_dir) + os.pathsep + self.env["PATH"])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(self.capture.read_text().splitlines()), 4)
        self.assertIn("training=0 tee=9", result.stdout)

    def test_preview_never_launches_training(self):
        self.env.pop("PBS_JOBID")
        result = self.run_suite("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.capture.exists())
        self.assertFalse((self.root / "logs").exists())
        self.assertIn("local_", result.stdout)
        self.assertEqual(result.stdout.count("[Experiment]"), 4)

    def test_bad_arguments_fail_before_training(self):
        for args in [("--unknown",), ("--dry-run", "extra")]:
            result = self.run_suite(*args)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(self.capture.exists())


if __name__ == "__main__":
    unittest.main()