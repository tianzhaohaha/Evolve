"""CPU-only checks for run_ours_method.sh / run_ours_suite.sh with a fake launcher, including a drift guard
between the method launcher and arm E5 of run_ours_debug.sh (both read _common/ours_method.sh)."""

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
FILES = ("run_ours_suite.sh", "run_ours_suite_node1.sh", "run_ours_suite_node2.sh", "run_ours_method.sh", "run_ours_debug.sh",
         "_common/ours_method.sh", "agentstream_full.env")
NODE_JOBS = {"run_ours_suite_node1.sh": ["isolated", "interleaved"], "run_ours_suite_node2.sh": ["isolated"]}  # S1,S2 / S3
VERSION = re.search(r"AGENTSTREAM_RUN_VERSION=\$\{AGENTSTREAM_RUN_VERSION:-(\w+)\}", (SEED_ROOT / RUNNER_DIR / "agentstream_full.env").read_text()).group(1)
# job id, model, mode, experiment prefix (the baseline suite's naming: <method>_<model tag>_..._b10_steps<N>)
SETTINGS = [("S1", "Qwen3-4B-Instruct-2507", "isolated", f"ours_qwen3_4b_2507_agentstream_{VERSION}_n64_single_pass_b10_steps7"),
            ("S2", "Qwen2.5-7B-Instruct", "interleaved", f"ours_qwen25_7b_agentstream_{VERSION}_n64_single_pass_b10_steps20"),
            ("S3", "Qwen2.5-7B-Instruct", "isolated", f"ours_qwen25_7b_agentstream_{VERSION}_n64_single_pass_b10_steps7")]
METHOD = {  # the E5 configuration every run must carry
    "AGENTSTREAM_SEED_OPD_LOSS_COEF": "0.01", "AGENTSTREAM_SEED_OPD_GEN_LOSS_COEF": "0",
    "AGENTSTREAM_SEED_SUCCESS_ONLY": "True", "AGENTSTREAM_SEED_OPD_NORM_MODE": "response", "AGENTSTREAM_SEED_TRAJ_GAP_GATE": "True",
    "AGENTSTREAM_SEED_SIBLING_RESAMPLE": "True", "AGENTSTREAM_SEED_SIBLING_RESAMPLE_MAX_GROUPS": "4", "AGENTSTREAM_SEED_SIBLING_RESAMPLE_BASELINE": "source",
    "AGENTSTREAM_SEED_SIBLING_RESAMPLE_POOL_MAX_GROUPS": "3", "AGENTSTREAM_SEED_GLOBAL_POOL_SOURCE": "pool", "AGENTSTREAM_SEED_GLOBAL_POOL_ADMISSION": "success",
    "AGENTSTREAM_SEED_GLOBAL_POOL_JUDGE_BACKEND": "policy_vllm", "AGENTSTREAM_SEED_GLOBAL_POOL_EVICT_POLICY": "window",
    "AGENTSTREAM_SEED_GLOBAL_POOL_WINDOW_STEPS": "10", "AGENTSTREAM_SEED_GLOBAL_POOL_REWRITE": "deinstantiate",
    "AGENTSTREAM_SEED_EMA_MODE": "off", "AGENTSTREAM_SEED_REPLAY_ENABLE": "False", "AGENTSTREAM_SEED_ROUTE_MODE": "none",
}


class OursMethodRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        scripts = Path(self.temp.name) / "SEED" / RUNNER_DIR
        (scripts / "_common").mkdir(parents=True)
        for filename in FILES:
            shutil.copyfile(SEED_ROOT / RUNNER_DIR / filename, scripts / filename)
        self.scripts = scripts
        self.capture = Path(self.temp.name) / "calls.jsonl"
        (scripts / "run_agentstream_sft_glm_self.sh").write_text(
            "#!/usr/bin/env bash\n"
            "python3 - \"$@\" <<'PY'\n"
            "import json, os, sys\n"
            "with open(os.environ['CAPTURE'], 'a') as f:\n"
            "    f.write(json.dumps({'argv': sys.argv[1:], 'env': dict(os.environ)}) + '\\n')\n"
            "PY\n"
            '[[ -n "${FAIL_MODEL:-}" && "$AGENTSTREAM_BASE_MODEL_NAME" == "$FAIL_MODEL" ]] && exit 7\n'
            "exit 0\n"
        )
        self.env = {"PATH": os.environ["PATH"], "HOME": self.temp.name, "CAPTURE": str(self.capture)}

    def run_script(self, script, *args, **env):
        return subprocess.run(
            ["bash", str(self.scripts / script), *args], cwd=self.temp.name,
            env={**self.env, **env}, capture_output=True, text=True,
        )

    def calls(self):
        return [json.loads(line) for line in self.capture.read_text().splitlines()]

    def test_suite_runs_the_three_settings_with_the_method_switches(self):
        result = self.run_script("run_ours_suite.sh")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 3)
        for call, (job, model, mode, prefix) in zip(calls, SETTINGS):
            argv, env = call["argv"], call["env"]
            self.assertEqual(argv, [mode, "trainer.resume_mode=auto"], job)
            self.assertEqual(env["AGENTSTREAM_BASE_MODEL_NAME"], model, job)
            self.assertEqual(env["AGENTSTREAM_METHOD_TAG"], "ours", job)
            self.assertEqual(env["AGENTSTREAM_EXPERIMENT_PREFIX"], prefix, job)
            self.assertEqual((env["AGENTSTREAM_BENCHMARKS"], env["AGENTSTREAM_RL_STREAM_PROFILE"], env["AGENTSTREAM_RL_TRAIN_DATA_SIZE"]), ("bfcl,tau2,browsecompplus", "single_pass", "10"), job)
            self.assertEqual(env["AGENTSTREAM_RL_SAVE_FREQ"], "0", job)
            for key in ("AGENTSTREAM_RL_EPOCHS", "AGENTSTREAM_RL_ISOLATED_EPOCHS", "AGENTSTREAM_MODEL_TAG", "EXPERIMENT_NAME"):
                self.assertNotIn(key, env, f"{job}: {key} must derive from agentstream_full.env")
            for key, value in METHOD.items():
                self.assertEqual(env.get(key), value, f"{job}: {key}")

    def test_inherited_model_and_step_variables_are_cleared_and_benchmarks_may_be_narrowed(self):
        leaked = {"AGENTSTREAM_SFT_MODEL_DIR": "/elsewhere/sft", "AGENTSTREAM_BASE_MODEL_PATH": "/elsewhere/base", "AGENTSTREAM_MODEL_TAG": "wrong",
                  "AGENTSTREAM_RL_EPOCHS": "3", "AGENTSTREAM_RL_ISOLATED_EPOCHS": "2", "AGENTSTREAM_EXPERIMENT_PREFIX": "stale"}
        result = self.run_script("run_ours_suite.sh", OURS_JOBS="S3", AGENTSTREAM_BENCHMARKS="tau2", **leaked)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        (call,) = self.calls()
        for key in leaked:
            self.assertNotEqual(call["env"].get(key), leaked[key], key)
        self.assertEqual(call["env"]["AGENTSTREAM_EXPERIMENT_PREFIX"], SETTINGS[2][3])  # steps7, not 2
        self.assertEqual(call["env"]["AGENTSTREAM_BENCHMARKS"], "tau2")

    def test_node_wrappers_split_the_settings(self):
        for script, modes in NODE_JOBS.items():
            self.capture.unlink(missing_ok=True)
            result = self.run_script(script)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = self.calls()
            self.assertEqual([c["argv"][0] for c in calls], modes, script)
            self.assertEqual(result.stdout.count("[SKIP]"), 3 - len(modes), script)
        self.assertEqual([c["env"]["AGENTSTREAM_BASE_MODEL_NAME"] for c in calls], ["Qwen2.5-7B-Instruct"])  # node2 = S3
        self.capture.unlink()
        self.run_script("run_ours_suite_node1.sh", OURS_JOBS="S2")  # the environment still wins
        self.assertEqual([c["argv"][0] for c in self.calls()], ["interleaved"])

    def test_subset_save_freq_and_failure_handling(self):
        result = self.run_script("run_ours_suite.sh", OURS_JOBS="S3,S2", OURS_SAVE_FREQ="10", FAIL_MODEL="Qwen2.5-7B-Instruct")
        self.assertEqual(result.returncode, 1)
        calls = self.calls()
        self.assertEqual([c["argv"][0] for c in calls], ["interleaved", "isolated"])  # script order
        self.assertTrue(all(c["env"]["AGENTSTREAM_RL_SAVE_FREQ"] == "10" for c in calls))
        self.assertIn("[SKIP] S1", result.stdout)
        self.assertIn("[FAILED] S2", result.stdout)
        self.assertIn("[FAILED] S3", result.stdout)

    def test_method_launcher_matches_arm_e5_of_the_debug_script(self):
        # The method must be exactly the arm that measured +4.8: every AGENTSTREAM_SEED_* value equal.
        self.run_script("run_ours_method.sh", "interleaved", "trainer.resume_mode=auto")
        (method,) = self.calls()
        self.capture.unlink()
        self.run_script("run_ours_debug.sh", DEBUG_ARMS="E5")
        (arm,) = self.calls()
        seed = lambda env: {k: v for k, v in env.items() if k.startswith("AGENTSTREAM_SEED_")}
        self.assertEqual(seed(method["env"]), seed(arm["env"]))
        self.assertEqual(method["argv"], ["interleaved", "trainer.resume_mode=auto"])
        self.assertEqual(method["env"]["AGENTSTREAM_METHOD_TAG"], "ours")
        self.assertNotIn("AGENTSTREAM_EXPERIMENT_PREFIX", method["env"])  # the name follows the suite convention
