# Copyright 2026 SEED x AgentStream integration.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ray-vectorized ALFWorld with per-reset designated game files.

Unlike env_package/alfworld (whose internal iterator picks games), each worker
here binds to an explicit game file on reset — the hook that lets a
TaskStreamScheduler impose Isolated / Sequential / Interleaved orderings over
the 6 ALFWorld task types. Registration-per-reset follows the
FixedGameFileBatchEnv precedent in scripts/sft/_common/pipeline.py.

Return shapes match env_package/alfworld/envs.py::AlfworldEnvs so the existing
AlfWorldEnvironmentManager works unchanged (text_obs, image_obs=None, infos
with 'won' / 'extra.gamefile' / admissible commands).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import ray
import yaml

from agent_system.environments.env_package.agentstream.task_stream import (
    StreamBatch,
    ValTaskCycler,
)

TASK_TYPES = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]


def collect_game_files(alf_config_path: str) -> Dict[str, List[str]]:
    """Walk the ALFWorld train split and group solvable game files by task type."""
    import json

    with open(alf_config_path) as f:
        config = yaml.safe_load(f)
    data_path = os.path.expandvars(config["dataset"]["data_path"])
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"ALFWorld train data path not found: {data_path}. Check ALFWORLD_DATA."
        )

    grouped: Dict[str, List[str]] = {task_type: [] for task_type in TASK_TYPES}
    for root, _dirs, files in os.walk(data_path, topdown=False):
        if "game.tw-pddl" not in files or "movable" in root or "Sliced" in root:
            continue
        game_path = os.path.join(root, "game.tw-pddl")
        try:
            with open(game_path, encoding="utf-8") as f:
                if not json.load(f).get("solvable", False):
                    continue
        except Exception:
            continue
        # pick_and_place_simple sorts before pick_and_place task types below.
        text = root
        task_type = next((t for t in TASK_TYPES if t in text), None)
        if task_type is None and "pick_and_place_simple" in text:
            task_type = "pick_and_place"
        if task_type is not None:
            grouped[task_type].append(game_path)

    for files_list in grouped.values():
        files_list.sort()
    return grouped


class AlfWorldStreamWorker:
    """Ray actor holding one single-game textworld env, rebound on each reset."""

    def __init__(self, alf_config_path: str, seed: int):
        with open(alf_config_path) as f:
            self.config = yaml.safe_load(f)
        self.seed = seed
        self.env = None
        self._done = True
        self._admissible: List[str] = []

    def _make_env(self, game_file: str):
        import textworld
        import textworld.gym
        from agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env import (
            AlfredDemangler,
            AlfredInfos,
        )

        wrappers = [
            AlfredDemangler(shuffle=bool(self.config["env"].get("domain_randomization", False))),
            AlfredInfos,
        ]
        request_infos = textworld.EnvInfos(
            won=True, admissible_commands=True, extras=["gamefile"]
        )
        max_steps = self.config["rl"]["training"]["max_nb_steps_per_episode"]
        env_id = textworld.gym.register_games(
            [game_file],
            request_infos,
            batch_size=1,
            asynchronous=False,
            max_episode_steps=max_steps,
            wrappers=wrappers,
        )
        return textworld.gym.make(env_id)

    @staticmethod
    def _info_at(infos: Any, idx: int = 0) -> Dict[str, Any]:
        if isinstance(infos, list):
            return dict(infos[idx])
        item: Dict[str, Any] = {}
        for key, value in (infos or {}).items():
            item[key] = value[idx] if isinstance(value, (list, tuple)) else value
        return item

    def reset(self, game_file: str) -> Tuple[str, Dict[str, Any]]:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.env = self._make_env(game_file)
        if hasattr(self.env, "seed"):
            self.env.seed(self.seed)
        obs, infos = self.env.reset()
        info = self._info_at(infos)
        info.setdefault("extra.gamefile", game_file)
        info["won"] = bool(info.get("won", False))
        self._admissible = list(info.get("admissible_commands", []))
        self._done = False
        return str(obs[0]), info

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        if self._done or self.env is None:
            return "", 0.0, True, {"won": False, "post_done": True, "admissible_commands": []}
        obs, _scores, dones, infos = self.env.step([action])
        info = self._info_at(infos)
        won = bool(info.get("won", False))
        info["won"] = won
        done = bool(dones[0]) if isinstance(dones, (list, tuple)) else bool(dones)
        self._done = done
        self._admissible = list(info.get("admissible_commands", []))
        return str(obs[0]), 10.0 * float(won), done, info

    def get_admissible(self) -> List[str]:
        return self._admissible

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass


class AlfWorldStreamEnvs:
    """Vectorized stream-controlled ALFWorld, API-compatible with AlfworldEnvs."""

    def __init__(
        self,
        alf_config_path: str,
        task_source: Any,  # TaskStreamScheduler | ValTaskCycler
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: Dict[str, Any],
    ):
        self.task_source = task_source
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self.multi_modal = False
        self.last_batch_meta: Optional[StreamBatch] = None
        self.prev_admissible_commands: List[List[str]] = [[] for _ in range(self.num_processes)]

        if not ray.is_initialized():
            ray.init()
        worker_cls = ray.remote(**resources_per_worker)(AlfWorldStreamWorker)
        self.workers = [
            worker_cls.remote(alf_config_path, seed + (i // group_n))
            for i in range(self.num_processes)
        ]

    def reset(self):
        if isinstance(self.task_source, ValTaskCycler):
            self.task_source.reset_cursor()
        batch = self.task_source.next_batch(self.env_num).replicate(self.group_n)
        self.last_batch_meta = batch

        results = ray.get(
            [
                worker.reset.remote(batch.refs[i][1])  # (task_type, game_file)
                for i, worker in enumerate(self.workers)
            ]
        )
        text_obs, infos = [], []
        for i, (obs, info) in enumerate(results):
            self.prev_admissible_commands[i] = list(info.get("admissible_commands", []))
            text_obs.append(obs)
            infos.append(info)
        return text_obs, None, infos

    def step(self, actions: List[str]):
        assert len(actions) == self.num_processes
        results = ray.get(
            [worker.step.remote(actions[i]) for i, worker in enumerate(self.workers)]
        )
        text_obs, rewards, dones, infos = [], [], [], []
        for i, (obs, reward, done, info) in enumerate(results):
            self.prev_admissible_commands[i] = list(info.get("admissible_commands", []))
            text_obs.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return text_obs, None, rewards, dones, infos

    @property
    def get_admissible_commands(self) -> List[List[str]]:
        return self.prev_admissible_commands

    def close(self) -> None:
        try:
            ray.get([worker.close.remote() for worker in self.workers])
        except Exception:
            pass
