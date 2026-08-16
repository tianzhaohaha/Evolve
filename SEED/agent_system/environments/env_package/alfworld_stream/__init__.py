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

"""Stream-controlled ALFWorld: the 6 task types act as domains.

Applies the Isolated / Sequential / Interleaved streaming scenarios (and the
SEED-original ``random`` control) to SEED's existing ALFWorld benchmark, with
zero dependency conflicts. This is the cheap controlled counterpart to the
AgentStream suite: same scheduler, same online-metrics schema, same launcher
conventions — only the task source differs (train-split game files grouped by
task type instead of exgentic benchmarks).

Entry point: :func:`factory.make_alfworld_stream_envs` via
``env.env_name=alfworld_stream/AlfredTWEnv``.
"""

from .factory import make_alfworld_stream_envs

__all__ = ["make_alfworld_stream_envs"]
