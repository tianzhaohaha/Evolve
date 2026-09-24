"""CPU-only checks for run_ours_debug.sh and its node wrappers with a fake launcher."""

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
SCRIPTS = ("run_ours_debug.sh", "run_ours_debug_node1.sh", "run_ours_debug_node2.sh", "run_ours_debug_node3.sh")
ARMS = ["E1", "E3", "E4", "E5", "E6", "E2a", "E2b", "E2c", "E7", "E8a", "E8b", "E8c"]
# node -> (anchor tags that must run first, arms)
NODE_ARMS = {"run_ours_debug_node1.sh": (["a3"], ["E1", "E3", "E4", "E2a"]), "run_ours_debug_node2.sh": (["a3"], ["E1", "E5", "E6", "E2b"]),
             "run_ours_debug_node3.sh": (["a3", "a4"], ["E7", "E8a", "E8b", "E8c", "E1", "E2c"])}
FORMS = (("raw", "none"), ("deinst", "deinstantiate"), ("agg", "aggregate"))
A4 = ("True", "response", "False")  # success_only, opd_norm_mode, traj_gap_gate


class OursDebugRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        scripts = Path(self.temp.name) / "SEED" / RUNNER_DIR
        scripts.mkdir(parents=True)
        (scripts / "_common").mkdir()
        for filename in SCRIPTS + ("_common/ours_method.sh",):
            shutil.copyfile(SEED_ROOT / RUNNER_DIR / filename, scripts / filename)
        self.scripts = scripts
        self.capture = Path(self.temp.name) / "calls.jsonl"
        # Fake launcher: record argv + environment; FAIL_TAG makes the arm with that tag exit 7.
        (scripts / "run_agentstream_sft_glm_self.sh").write_text(
            "#!/usr/bin/env bash\n"
            "python3 - \"$@\" <<'PY'\n"
            "import json, os, sys\n"
            "with open(os.environ['CAPTURE'], 'a') as f:\n"
            "    f.write(json.dumps({'argv': sys.argv[1:], 'env': dict(os.environ)}) + '\\n')\n"
            "PY\n"
            '[[ -n "${FAIL_TAG:-}" && "$AGENTSTREAM_EXPERIMENT_PREFIX" == *"_${FAIL_TAG}" ]] && exit 7\n'
            "exit 0\n"
        )
        self.env = {"PATH": os.environ["PATH"], "HOME": self.temp.name, "CAPTURE": str(self.capture)}

    def run_script(self, script="run_ours_debug.sh", *args, **env):
        return subprocess.run(
            ["bash", str(self.scripts / script), *args], cwd=self.temp.name,
            env={**self.env, **env}, capture_output=True, text=True,
        )

    def calls(self):
        return [json.loads(line) for line in self.capture.read_text().splitlines()]

    @staticmethod
    def arm_of(call):
        return re.fullmatch(r"debug_ours_(?P<run>.+?)_(?P<tag>a[34](?:_[a-z_]+)?)", call["env"]["AGENTSTREAM_EXPERIMENT_PREFIX"])

    def test_every_arm_runs_with_a_kept_last_checkpoint_and_resume(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), len(ARMS))
        by_tag = {}
        for call in calls:
            argv, env = call["argv"], call["env"]
            match = self.arm_of(call)
            self.assertIsNotNone(match, env["AGENTSTREAM_EXPERIMENT_PREFIX"])
            self.assertRegex(match["run"], r"\d{8}_\d{6}")  # no PBS_JOBID -> timestamp; no step count in the name
            by_tag[match["tag"]] = env
            self.assertEqual(argv[:2], ["interleaved", "trainer.resume_mode=auto"])
            self.assertEqual(env["AGENTSTREAM_RL_EPOCHS"], "10")
            self.assertEqual(env["AGENTSTREAM_RL_SAVE_FREQ"], "10")
            self.assertEqual(env["AGENTSTREAM_RL_MAX_CKPT_TO_KEEP"], "1")
            self.assertEqual(env["ENV_FILE"], "/dev/null")
            self.assertNotIn("EXPERIMENT_NAME", env)
            # success_only + response norm everywhere (the gate only on the a3 family); the SEED baseline switches pinned
            expected_gate = "True" if match["tag"].startswith("a3") else "False"
            self.assertEqual((env["AGENTSTREAM_SEED_SUCCESS_ONLY"], env["AGENTSTREAM_SEED_OPD_NORM_MODE"], env["AGENTSTREAM_SEED_TRAJ_GAP_GATE"]), ("True", "response", expected_gate))
            self.assertEqual(env["AGENTSTREAM_SEED_OPD_LOSS_COEF"], "0.01")
        self.assertEqual(list(by_tag), ["a3", "a3_resample", "a3_resample_pool_raw", "a3_resample_pool_deinst", "a3_resample_pool_agg",
                                        "a3_genopd_raw", "a3_genopd_deinst", "a3_genopd_agg", "a4", "a4_genopd_raw", "a4_genopd_deinst", "a4_genopd_agg"])
        anchor = by_tag["a3"]
        self.assertEqual((anchor["AGENTSTREAM_SEED_SIBLING_RESAMPLE"], anchor["AGENTSTREAM_SEED_GLOBAL_POOL_SOURCE"], anchor["AGENTSTREAM_SEED_OPD_GEN_LOSS_COEF"]), ("False", "copy", "0"))
        self.assertEqual((anchor["AGENTSTREAM_SEED_SIBLING_RESAMPLE_BASELINE"], anchor["AGENTSTREAM_SEED_SIBLING_RESAMPLE_POOL_MAX_GROUPS"]), ("own", "0"))
        self.assertEqual((anchor["AGENTSTREAM_SEED_GLOBAL_POOL_ADMISSION"], anchor["AGENTSTREAM_SEED_GLOBAL_POOL_JUDGE_BACKEND"], anchor["AGENTSTREAM_SEED_GLOBAL_POOL_REWRITE"]), ("gap", "openai", "none"))
        resample = by_tag["a3_resample"]
        self.assertEqual((resample["AGENTSTREAM_SEED_SIBLING_RESAMPLE"], resample["AGENTSTREAM_SEED_SIBLING_RESAMPLE_BASELINE"], resample["AGENTSTREAM_SEED_GLOBAL_POOL_SOURCE"]), ("True", "source", "copy"))
        for form, rewrite in FORMS:
            env = by_tag[f"a3_resample_pool_{form}"]
            self.assertEqual((env["AGENTSTREAM_SEED_SIBLING_RESAMPLE"], env["AGENTSTREAM_SEED_SIBLING_RESAMPLE_POOL_MAX_GROUPS"], env["AGENTSTREAM_SEED_OPD_GEN_LOSS_COEF"]), ("True", "3", "0"))
            self.assertEqual((env["AGENTSTREAM_SEED_GLOBAL_POOL_SOURCE"], env["AGENTSTREAM_SEED_GLOBAL_POOL_ADMISSION"], env["AGENTSTREAM_SEED_GLOBAL_POOL_JUDGE_BACKEND"]), ("pool", "success", "policy_vllm"))
            self.assertEqual((env["AGENTSTREAM_SEED_GLOBAL_POOL_EVICT_POLICY"], env["AGENTSTREAM_SEED_GLOBAL_POOL_WINDOW_STEPS"], env["AGENTSTREAM_SEED_GLOBAL_POOL_REWRITE"]), ("window", "10", rewrite))
        base = by_tag["a4"]
        self.assertEqual((base["AGENTSTREAM_SEED_SIBLING_RESAMPLE"], base["AGENTSTREAM_SEED_GLOBAL_POOL_SOURCE"], base["AGENTSTREAM_SEED_OPD_GEN_LOSS_COEF"]), ("False", "copy", "0"))
        for prefix in ("a3", "a4"):  # gated (a3) and ungated (a4) gen OPD from the pool, same pool settings
            for form, rewrite in FORMS:
                env = by_tag[f"{prefix}_genopd_{form}"]
                self.assertEqual((env["AGENTSTREAM_SEED_SIBLING_RESAMPLE"], env["AGENTSTREAM_SEED_OPD_GEN_LOSS_COEF"], env["AGENTSTREAM_SEED_GLOBAL_POOL_SOURCE"]), ("False", "0.005", "pool"))
                self.assertEqual((env["AGENTSTREAM_SEED_GLOBAL_POOL_ADMISSION"], env["AGENTSTREAM_SEED_GLOBAL_POOL_JUDGE_BACKEND"], env["AGENTSTREAM_SEED_GLOBAL_POOL_REWRITE"]), ("success", "policy_vllm", rewrite))

    def test_debug_arms_filters_and_a_failed_arm_does_not_stop_the_others(self):
        result = self.run_script(DEBUG_ARMS="E4,E8b", FAIL_TAG="a3_resample_pool_raw")
        self.assertEqual(result.returncode, 1)
        self.assertEqual([self.arm_of(c)["tag"] for c in self.calls()], ["a3_resample_pool_raw", "a4_genopd_deinst"])
        self.assertIn("[SKIP] E1 A3 floor", result.stdout)
        self.assertIn("[FAILED] E4", result.stdout)
        self.assertIn("[SUCCESS] E8b", result.stdout)

    def test_checkpoint_settings_honour_the_environment(self):
        result = self.run_script(DEBUG_ARMS="E1", AGENTSTREAM_RL_SAVE_FREQ="0", AGENTSTREAM_RL_MAX_CKPT_TO_KEEP="2")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        (call,) = self.calls()
        self.assertEqual((call["env"]["AGENTSTREAM_RL_SAVE_FREQ"], call["env"]["AGENTSTREAM_RL_MAX_CKPT_TO_KEEP"]), ("0", "2"))

    def test_continuation_reuses_the_run_id_and_moves_the_checkpoint_to_the_new_last_step(self):
        result = self.run_script(DEBUG_RUN_ID="70000.pbs", DEBUG_ARMS="E4", TOTAL_STEPS="20")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        (call,) = self.calls()
        self.assertEqual(call["env"]["AGENTSTREAM_EXPERIMENT_PREFIX"], "debug_ours_70000.pbs_a3_resample_pool_raw")
        self.assertEqual((call["env"]["AGENTSTREAM_RL_EPOCHS"], call["env"]["AGENTSTREAM_RL_SAVE_FREQ"]), ("20", "20"))
        self.assertIn("trainer.resume_mode=auto", call["argv"])

    def test_node_wrappers_split_the_arms_and_each_family_keeps_its_anchor(self):
        for script, (anchors, arms) in NODE_ARMS.items():
            self.capture.unlink(missing_ok=True)
            result = self.run_script(script, PBS_JOBID="71000")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            tags = [self.arm_of(c)["tag"] for c in self.calls()]
            self.assertEqual(len(tags), len(arms))
            for anchor in anchors:  # every arm's anchor runs on the same node, before the arms that pair with it
                self.assertIn(anchor, tags)
                self.assertTrue(all(tags.index(anchor) < tags.index(t) for t in tags if t.startswith(anchor + "_")), tags)
            self.assertTrue(all(self.arm_of(c)["run"] == "71000" for c in self.calls()))
            self.assertEqual(result.stdout.count("[SKIP]"), len(ARMS) - len(arms))
        self.capture.unlink()
        self.run_script("run_ours_debug_node1.sh", DEBUG_ARMS="E1")  # the environment still wins
        self.assertEqual([self.arm_of(c)["tag"] for c in self.calls()], ["a3"])
