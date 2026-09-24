"""CPU-only checks for run_ours_final.sh (continuations / reruns of run_ours_debug.sh arms) with a fake launcher."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SEED_ROOT = Path(__file__).resolve().parents[2]
RUNNER_DIR = Path("examples/agentstream_trainer")
EXPECTED = {  # job id -> (experiment prefix, epochs, save_freq, keep, wandb resume id)
    "F1": ("debug_ours_71186.gaas_a3_resample_pool_deinst", "20", "20", "1", "bjnan8uy"),
    "F2": ("debug_ours_71185.gaas_a3_resample", "20", "20", "1", "dx7l9c78"),
    "F3": ("debug_ours_final_a3_resample", "20", "10", "2", None),
}


class OursFinalRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        scripts = Path(self.temp.name) / "SEED" / RUNNER_DIR
        scripts.mkdir(parents=True)
        (scripts / "_common").mkdir()
        for filename in ("run_ours_final.sh", "run_ours_debug.sh", "_common/ours_method.sh"):
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
            '[[ -n "${FAIL_PREFIX:-}" && "$AGENTSTREAM_EXPERIMENT_PREFIX" == "$FAIL_PREFIX" ]] && exit 7\n'
            "exit 0\n"
        )
        self.env = {"PATH": os.environ["PATH"], "HOME": self.temp.name, "CAPTURE": str(self.capture)}

    def run_script(self, *args, **env):
        return subprocess.run(
            ["bash", str(self.scripts / "run_ours_final.sh"), *args], cwd=self.temp.name,
            env={**self.env, **env}, capture_output=True, text=True,
        )

    def calls(self):
        return [json.loads(line) for line in self.capture.read_text().splitlines()]

    def test_runs_the_three_jobs_with_their_continuation_settings(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()
        self.assertEqual([c["env"]["AGENTSTREAM_EXPERIMENT_PREFIX"] for c in calls], [v[0] for v in EXPECTED.values()])
        for call, (job, (prefix, epochs, save_freq, keep, wandb_id)) in zip(calls, EXPECTED.items()):
            argv, env = call["argv"], call["env"]
            self.assertEqual(argv[:2], ["interleaved", "trainer.resume_mode=auto"], job)
            self.assertEqual((env["AGENTSTREAM_RL_EPOCHS"], env["AGENTSTREAM_RL_SAVE_FREQ"], env["AGENTSTREAM_RL_MAX_CKPT_TO_KEEP"]), (epochs, save_freq, keep), job)
            self.assertEqual(env.get("WANDB_RUN_ID"), wandb_id, job)
            self.assertEqual(env.get("WANDB_RESUME"), "allow" if wandb_id else None, job)
            # the arm's own switches come from run_ours_debug.sh unchanged
            self.assertEqual((env["AGENTSTREAM_SEED_SUCCESS_ONLY"], env["AGENTSTREAM_SEED_TRAJ_GAP_GATE"], env["AGENTSTREAM_SEED_SIBLING_RESAMPLE"], env["AGENTSTREAM_SEED_SIBLING_RESAMPLE_BASELINE"]), ("True", "True", "True", "source"), job)
        self.assertEqual(calls[0]["env"]["AGENTSTREAM_SEED_SIBLING_RESAMPLE_POOL_MAX_GROUPS"], "3")
        self.assertEqual((calls[1]["env"]["AGENTSTREAM_SEED_GLOBAL_POOL_SOURCE"], calls[2]["env"]["AGENTSTREAM_SEED_GLOBAL_POOL_SOURCE"]), ("copy", "copy"))

    def test_subset_and_failure_handling(self):
        result = self.run_script(FINAL_JOBS="F3,F1", FAIL_PREFIX="debug_ours_final_a3_resample")
        self.assertEqual(result.returncode, 1)
        self.assertEqual([c["env"]["AGENTSTREAM_EXPERIMENT_PREFIX"] for c in self.calls()],
                         ["debug_ours_71186.gaas_a3_resample_pool_deinst", "debug_ours_final_a3_resample"])  # script order, not list order
        self.assertIn("[SKIP] F2", result.stdout)
        self.assertIn("[FAILED] F3", result.stdout)
        self.assertIn("[SUCCESS] F1", result.stdout)

    def test_final_run_id_names_the_fresh_run(self):
        result = self.run_script(FINAL_JOBS="F3", FINAL_RUN_ID="final2")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        (call,) = self.calls()
        self.assertEqual(call["env"]["AGENTSTREAM_EXPERIMENT_PREFIX"], "debug_ours_final2_a3_resample")
        self.assertNotIn("WANDB_RUN_ID", call["env"])

    def test_dry_run_flag_reaches_the_launcher(self):
        result = self.run_script("--dry-run", FINAL_JOBS="F1")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.calls()[0]["env"].get("DRY_RUN"), "true")
