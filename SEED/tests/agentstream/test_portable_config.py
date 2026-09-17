"""CPU-only checks for checkout-relative configuration and local asset ignores.

Run directly with Python; no benchmark environments or credentials are loaded.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path("SEED/examples/agentstream_trainer/agentstream_full.env")
VARIABLES = (
    "AGENTSTREAM_REPO_ROOT", "AGENTSTREAM_SEED_ROOT", "AGENTSTREAM_EXGENTIC_ROOT",
    "AGENTSTREAM_LLMS_ROOT", "MODELS_ROOT", "CHECKPOINTS_ROOT",
    "AGENTSTREAM_BASE_MODEL_PATH", "AGENTSTREAM_SFT_MODEL_DIR",
    "AGENTSTREAM_SFT_DATA_DIR", "EMBED_MODEL_PATH", "TMPDIR", "RAY_TMPDIR",
    "AGENTSTREAM_RL_GPUS", "AGENTSTREAM_RL_N_GPUS", "AGENTSTREAM_SFT_GPUS",
    "AGENTSTREAM_SFT_NPROC", "AGENTSTREAM_POLICY_GPU", "AGENTSTREAM_RETRIEVER_DEVICE",
    "AGENTSTREAM_RL_TRAIN_DATA_SIZE", "AGENTSTREAM_RL_EPOCHS",
    "AGENTSTREAM_BENCHMARKS",
)


class PortableConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # A moved checkout with spaces, sourced from a different working directory.
        self.repo = Path(self.temp.name) / "new server" / "Evolve"
        self.config = self.repo / CONFIG_PATH
        self.config.parent.mkdir(parents=True)
        shutil.copyfile(REPO_ROOT / CONFIG_PATH, self.config)

    def resolve(self, **overrides):
        # Deliberately do not inherit .env, credentials, or stale derived settings.
        env = {"PATH": os.environ["PATH"], "HOME": self.temp.name, "LC_ALL": "C"}
        env.update(overrides)
        result = subprocess.run(
            ["bash", "--noprofile", "--norc", "-c",
             'set -euo pipefail; source "$1"; shift; '
             'for key in "$@"; do printf "%s\\0%s\\0" "$key" "${!key-}"; done',
             "config-test", str(self.config), *VARIABLES],
            cwd=self.temp.name, env=env, capture_output=True, text=True, check=True,
        )
        fields = result.stdout.rstrip("\0").split("\0")
        return dict(zip(fields[::2], fields[1::2]))

    def test_relocated_checkout_defaults(self):
        values = self.resolve()
        self.assertEqual(values["AGENTSTREAM_REPO_ROOT"], str(self.repo))
        self.assertEqual(values["AGENTSTREAM_SEED_ROOT"], str(self.repo / "SEED"))
        self.assertEqual(values["AGENTSTREAM_EXGENTIC_ROOT"], str(self.repo / "AgentStream/exgentic"))
        self.assertEqual(values["AGENTSTREAM_LLMS_ROOT"], str(self.repo / "LLMs"))
        self.assertEqual(values["MODELS_ROOT"], str(self.repo / "LLMs"))
        self.assertEqual(values["CHECKPOINTS_ROOT"], str(self.repo / "ckpt"))
        self.assertEqual(values["AGENTSTREAM_BASE_MODEL_PATH"], str(self.repo / "LLMs/Qwen3-4B-Instruct-2507"))
        self.assertTrue(values["AGENTSTREAM_SFT_MODEL_DIR"].startswith(str(self.repo / "LLMs") + "/"))
        self.assertTrue(values["AGENTSTREAM_SFT_DATA_DIR"].startswith(str(self.repo / "SEED/outputs") + "/"))
        self.assertEqual(values["TMPDIR"], "")

    def test_two_gpus_and_unchanged_benchmarks(self):
        values = self.resolve()
        for key in ("AGENTSTREAM_RL_GPUS", "AGENTSTREAM_SFT_GPUS", "AGENTSTREAM_POLICY_GPU"):
            self.assertEqual(values[key], "0,1")
        for key in ("AGENTSTREAM_RL_N_GPUS", "AGENTSTREAM_SFT_NPROC", "AGENTSTREAM_RL_TRAIN_DATA_SIZE"):
            self.assertEqual(values[key], "2")
        self.assertEqual(values["AGENTSTREAM_RETRIEVER_DEVICE"], "cpu")
        self.assertEqual(values["AGENTSTREAM_RL_EPOCHS"], "75")  # 3 default benchmarks x 50 tasks / 2 per step
        self.assertEqual(values["AGENTSTREAM_BENCHMARKS"], "bfcl,tau2,browsecompplus")

    def test_explicit_storage_overrides(self):
        values = self.resolve(AGENTSTREAM_LLMS_ROOT="/storage/models", CHECKPOINTS_ROOT="/storage/ckpt")
        self.assertEqual(values["MODELS_ROOT"], "/storage/models")
        self.assertEqual(values["CHECKPOINTS_ROOT"], "/storage/ckpt")
        self.assertEqual(self.resolve(MODELS_ROOT="/legacy/models")["AGENTSTREAM_LLMS_ROOT"], "/legacy/models")

    def test_local_embedding_detection(self):
        embedding = self.repo / "LLMs/Qwen3-Embedding-8B"
        embedding.mkdir(parents=True)
        (embedding / "config.json").write_text("{}")
        self.assertEqual(self.resolve()["EMBED_MODEL_PATH"], str(embedding))
        self.assertEqual(self.resolve(EMBED_MODEL_PATH="/override/model")["EMBED_MODEL_PATH"], "/override/model")

    def test_scratch_override_respects_job_environment(self):
        scratch = str(Path(self.temp.name) / "scratch")
        values = self.resolve(AGENTSTREAM_TMP_ROOT=scratch)
        self.assertEqual(values["TMPDIR"], scratch)
        self.assertEqual(values["RAY_TMPDIR"], scratch + "/ray")
        values = self.resolve(AGENTSTREAM_TMP_ROOT=scratch, TMPDIR="/job/tmp", RAY_TMPDIR="/job/ray")
        self.assertEqual(values["TMPDIR"], "/job/tmp")
        self.assertEqual(values["RAY_TMPDIR"], "/job/ray")

    def test_git_ignores_local_assets_not_public_config(self):
        for path, ignored in (
            ("LLMs/model/config.json", True), ("ckpt/run/metadata.json", True),
            ("SEED/.env", True), ("SEED/.env.newserver", True),
            ("AgentStream/exgentic/.env.local", True),
            ("SEED/.env.example", False), (str(CONFIG_PATH), False),
        ):
            with self.subTest(path=path):
                result = subprocess.run(
                    ["git", "check-ignore", "--no-index", "-q", path],
                    cwd=REPO_ROOT, capture_output=True,
                )
                self.assertEqual(result.returncode, 0 if ignored else 1)


if __name__ == "__main__":
    unittest.main()