# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss

    DEDUP_AVAILABLE = True
except ImportError:
    DEDUP_AVAILABLE = False
    np = None  

from .playbook_utils import parse_playbook_line, format_playbook_line


class BulletpointAnalyzer:

    def __init__(
        self,
        llm_merge_fn: Optional[Any] = None,
        embedding_model_name: str = "all-mpnet-base-v2",
    ) -> None:
        self.llm_merge_fn = llm_merge_fn
        self.embedding_model_name = embedding_model_name
        self._embedding_model: Optional[Any] = None

    def _load_embedding_model(self) -> None:
        if self._embedding_model is None and DEDUP_AVAILABLE:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)

    @staticmethod
    def _parse_playbook(
        playbook: str,
    ) -> Tuple[List[str], List[Dict[str, Any]], Dict[int, int]]:
        lines = playbook.strip().split("\n")
        bullets: List[Dict[str, Any]] = []
        bullet_line_mapping: Dict[int, int] = {}

        for line_idx, line in enumerate(lines):
            parsed = parse_playbook_line(line)
            if parsed:
                parsed["line_number"] = line_idx + 1
                parsed["original_line"] = line
                bullet_index = len(bullets)
                bullet_line_mapping[bullet_index] = line_idx
                bullets.append(parsed)

        return lines, bullets, bullet_line_mapping

    def _compute_embeddings(self, bullets: List[Dict[str, Any]]) -> Any:
        if not DEDUP_AVAILABLE:
            raise RuntimeError("Cannot compute embeddings without sentence-transformers")
        self._load_embedding_model()
        contents = [b["content"] for b in bullets]
        embeddings = self._embedding_model.encode(
            contents, convert_to_numpy=True, show_progress_bar=False
        )
        faiss.normalize_L2(embeddings)
        return embeddings

    @staticmethod
    def _find_similar_groups(
        bullets: List[Dict[str, Any]],
        embeddings: Any,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        similarity_matrix = np.dot(embeddings, embeddings.T) 
        duplicate_groups: List[Dict[str, Any]] = []
        visited: set[int] = set()

        for i in range(len(bullets)):
            if i in visited:
                continue
            similar_indices = []
            for j in range(i + 1, len(bullets)):
                if similarity_matrix[i, j] >= threshold:
                    similar_indices.append(j)
            if similar_indices:
                group = [i] + similar_indices
                duplicate_groups.append(
                    {"indices": group, "bullets": [bullets[idx] for idx in group]}
                )
                visited.update(group)

        return duplicate_groups

    def _merge_bullets_with_llm(
        self, bullets_group: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if len(bullets_group) == 1:
            return bullets_group[0]

        if self.llm_merge_fn is None:
            return bullets_group[0]

        bullets_text = "\n".join(
            f"{i+1}. [{b['id']}] helpful={b['helpful']} harmful={b['harmful']} :: {b['content']}"
            for i, b in enumerate(bullets_group)
        )
        total_helpful = sum(b["helpful"] for b in bullets_group)
        total_harmful = sum(b["harmful"] for b in bullets_group)
        base_id = bullets_group[0]["id"]

        prompt = (
            f"You are merging similar playbook bulletpoints into a single, "
            f"comprehensive entry.\n\n"
            f"Given these similar bulletpoints:\n{bullets_text}\n\n"
            f"Merge them into ONE bulletpoint that captures all important "
            f"information while removing redundancy.\n\n"
            f"Requirements:\n"
            f"1. Keep the ID from the first entry: [{base_id}]\n"
            f"2. Use combined counts: helpful={total_helpful} harmful={total_harmful}\n"
            f"3. Combine the content to be comprehensive but concise\n"
            f"4. Output ONLY in this format: [{base_id}] helpful={total_helpful} "
            f"harmful={total_harmful} :: [merged content]\n\n"
            f"Do NOT include any explanation, just output the merged bulletpoint."
        )

        try:
            merged_content = self.llm_merge_fn(prompt).strip()
            pattern = r"\[([^\]]+)\]\s+helpful=(\d+)\s+harmful=(\d+)\s+::\s+(.+)"
            match = re.match(pattern, merged_content)
            if match:
                bullet_id, helpful, harmful, content = match.groups()
                return {
                    "id": bullet_id,
                    "helpful": int(helpful),
                    "harmful": int(harmful),
                    "content": content.strip(),
                    "original_line": format_playbook_line(
                        bullet_id, int(helpful), int(harmful), content.strip()
                    ),
                    "is_merged": True,
                    "original_count": len(bullets_group),
                }
            else:
                return bullets_group[0]
        except Exception:
            return bullets_group[0]

    def analyze(
        self,
        playbook: str,
        threshold: float = 0.90,
        merge: bool = True,
    ) -> str:
        if not DEDUP_AVAILABLE:
            return playbook

        original_lines, bullets, bullet_line_mapping = self._parse_playbook(playbook)

        if len(bullets) == 0:
            return playbook

        embeddings = self._compute_embeddings(bullets)
        duplicate_groups = self._find_similar_groups(bullets, embeddings, threshold)

        if len(duplicate_groups) == 0:
            return playbook

        merge_mapping: Dict[int, Dict[str, Any]] = {}
        processed_indices: set[int] = set()

        if merge:
            for group in duplicate_groups:
                indices = group["indices"]
                merged_bullet = self._merge_bullets_with_llm(group["bullets"])
                if merged_bullet:
                    merge_mapping[indices[0]] = merged_bullet
                    processed_indices.update(indices)
        else:
            for group in duplicate_groups:
                indices = group["indices"]
                processed_indices.update(indices[1:])

        output_lines: List[str] = []
        for line_idx, original_line in enumerate(original_lines):
            current_bullet_idx = None
            for bi, li in bullet_line_mapping.items():
                if li == line_idx:
                    current_bullet_idx = bi
                    break

            if current_bullet_idx is not None:
                if current_bullet_idx in merge_mapping:
                    output_lines.append(merge_mapping[current_bullet_idx]["original_line"])
                elif current_bullet_idx in processed_indices:
                    continue
                else:
                    output_lines.append(original_line)
            else:
                output_lines.append(original_line)

        return "\n".join(output_lines)
