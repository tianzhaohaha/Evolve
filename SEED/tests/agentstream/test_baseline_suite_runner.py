"""CPU-only checks for run_baseline_suite.sh with a fake baseline launcher."""

import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SEED_ROOT = Path(__file__).resolve().parents[2]
RUNNER_DIR = Path("examples/agentstream_trainer")
BASELINES = ["vanilla", "grpo", "seed", "opsd", "rlsd"]  # formal set; grpo_base / sdar stay available via run_agentstream_baseline.sh


class BaselineSuiteRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "SEED"
        scripts = self.root / RUNNER_DIR
        scripts.mkdir(parents=True)
        for filename in ("run_baseline_suite.sh", "agentstream_full.env"):
            shutil.copyfile(SEED_ROOT / RUNNER_DIR / filename, scripts / filename)
        self.script = scripts / "run_baseline_suite.sh"
        self.capture = Path(self.temp.name) / "calls.jsonl"
        self.ckpt = Path(self.temp.name) / "ckpt"
        (self.root / ".env").write_text(f"export CHECKPOINTS_ROOT={self.ckpt}\n")
        # Fake launcher: record argv + environment; FAIL_BASELINE simulates a crash.
        (scripts / "run_agentstream_baseline.sh").write_text(
            "#!/usr/bin/env bash\n"
            "python3 - \"$@\" <<'PY'\n"
            "import json, os, sys\n"
            "with open(os.environ['CAPTURE'], 'a') as f:\n"
            "    f.write(json.dumps({'argv': sys.argv[1:], 'env': dict(os.environ)}) + '\\n')\n"
            "PY\n"
            '[[ "$1" == "${FAIL_BASELINE:-}" ]] && exit 7\n'
            "exit 0\n"
        )
        self.env = {"PATH": os.environ["PATH"], "HOME": self.temp.name, "CAPTURE": str(self.capture)}

    def run_suite(self, *args, **env):
        return subprocess.run(
            ["bash", str(self.script), *args], cwd=self.temp.name,
            env={**self.env, **env}, capture_output=True, text=True,
        )

    def calls(self):
        return [json.loads(line) for line in self.capture.read_text().splitlines()]

    def test_runs_every_baseline_with_shared_settings(self):
        result = self.run_suite()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()
        self.assertEqual([c["argv"][0] for c in calls], BASELINES)
        for call in calls:
            argv, env = call["argv"], call["env"]
            self.assertEqual(argv[1], "interleaved")
            self.assertIn("trainer.resume_mode=auto", argv)
            self.assertEqual(env["AGENTSTREAM_BENCHMARKS"], "bfcl,tau2,browsecompplus")
            self.assertEqual(env["AGENTSTREAM_RL_TRAIN_DATA_SIZE"], "10")
            self.assertEqual(env["AGENTSTREAM_RL_EPOCHS"], "20")  # ceil(3 x 64 / 10): one single pass
            self.assertEqual(env["ENV_FILE"], "/dev/null")
            self.assertTrue(env["AGENTSTREAM_EXPERIMENT_PREFIX"].startswith(argv[0] + "_"))
            self.assertNotIn("EXPERIMENT_NAME", env)

    def test_skips_completed_and_continues_after_failure(self):
        # Experiment name follows agentstream_full.env; read the run version from it rather than hard-coding.
        env_text = (SEED_ROOT / RUNNER_DIR / "agentstream_full.env").read_text()
        version = re.search(r"AGENTSTREAM_RUN_VERSION=\$\{AGENTSTREAM_RUN_VERSION:-(\w+)\}", env_text).group(1)
        exp = f"grpo_qwen3_4b_2507_agentstream_{version}_n64_single_pass_b10_steps20_interleaved_online_s44"
        (self.ckpt / exp).mkdir(parents=True)
        (self.ckpt / exp / "latest_checkpointed_iteration.txt").write_text("20\n")
        result = self.run_suite(FAIL_BASELINE="seed")
        self.assertEqual(result.returncode, 1)
        self.assertEqual([c["argv"][0] for c in self.calls()], [b for b in BASELINES if b != "grpo"])
        self.assertIn("[SKIP] grpo", result.stdout)
        self.assertIn("Failed baseline: seed", result.stdout)

    def test_isolated_mode_runs_per_benchmark_and_skips_only_when_all_runs_finished(self):
        env_text = (SEED_ROOT / RUNNER_DIR / "agentstream_full.env").read_text()
        version = re.search(r"AGENTSTREAM_RUN_VERSION=\$\{AGENTSTREAM_RUN_VERSION:-(\w+)\}", env_text).group(1)
        exp = f"grpo_qwen3_4b_2507_agentstream_{version}_n64_single_pass_b10_steps7_isolated_online_s44"
        for bench in ("bfcl", "tau2"):  # browsecompplus run missing -> grpo must still run
            (self.ckpt / f"{exp}_{bench}").mkdir(parents=True)
            (self.ckpt / f"{exp}_{bench}" / "latest_checkpointed_iteration.txt").write_text("7\n")
        result = self.run_suite(STREAM_MODE="isolated")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()
        self.assertEqual([c["argv"][0] for c in calls], BASELINES)
        for call in calls:
            self.assertEqual(call["argv"][1], "isolated")
            self.assertEqual(call["env"]["AGENTSTREAM_RL_ISOLATED_EPOCHS"], "7")  # ceil(64 / 10) per benchmark
            self.assertTrue(call["env"]["AGENTSTREAM_EXPERIMENT_PREFIX"].endswith("_b10_steps7"))

        (self.ckpt / f"{exp}_browsecompplus").mkdir()
        (self.ckpt / f"{exp}_browsecompplus" / "latest_checkpointed_iteration.txt").write_text("7\n")
        self.capture.unlink()
        result = self.run_suite(STREAM_MODE="isolated")
        self.assertIn("[SKIP] grpo", result.stdout)
        self.assertEqual([c["argv"][0] for c in self.calls()], [b for b in BASELINES if b != "grpo"])
