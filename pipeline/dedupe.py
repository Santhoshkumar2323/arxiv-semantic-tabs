from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from pipeline.fetch import PaperRecord
from pipeline.logger import get_logger

logger = get_logger()

@dataclass
class DropRecord:
    arxiv_id: str
    title: str
    reason: str
    kept_arxiv_id: Optional[str] = None  


def dedupe_by_id(records: List[PaperRecord]) -> Tuple[List[PaperRecord], List[DropRecord]]:
    kept, dropped, _ = _dedupe_by_id_with_indices(records)
    return kept, dropped


def _dedupe_by_id_with_indices(
    records: List[PaperRecord],
) -> Tuple[List[PaperRecord], List[DropRecord], List[int]]:
    seen: dict[str, PaperRecord] = {}
    kept: List[PaperRecord] = []
    kept_indices: List[int] = []
    dropped: List[DropRecord] = []

    for idx, record in enumerate(records):
        if record.arxiv_id in seen:
            dropped.append(
                DropRecord(
                    arxiv_id=record.arxiv_id,
                    title=record.title,
                    reason="duplicate arXiv ID within sector pull",
                    kept_arxiv_id=seen[record.arxiv_id].arxiv_id,
                )
            )
            logger.debug(
                f"[{record.sector}] dropped '{record.title[:60]}' "
                f"({record.arxiv_id}): exact duplicate of an already-kept paper"
            )
        else:
            seen[record.arxiv_id] = record
            kept.append(record)
            kept_indices.append(idx)

    return kept, dropped, kept_indices


def _cosine_sim_matrix(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unstable_mask = (norms.flatten() < 1e-6) | ~np.isfinite(norms.flatten())

    if unstable_mask.any():
        logger.warning(
            f"cosine_sim_matrix: {int(unstable_mask.sum())} vector(s) had a "
            f"near-zero or non-finite norm — likely corrupted upstream text. "
            f"These are excluded from near-duplicate comparison, not auto-flagged either way."
        )

    safe_norms = norms.copy()
    safe_norms[unstable_mask.reshape(-1, 1)] = 1.0  # avoid dividing by near-zero
    normalized = vectors / safe_norms
    sim = normalized @ normalized.T
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)

    if unstable_mask.any():
        sim[unstable_mask, :] = 0.0
        sim[:, unstable_mask] = 0.0

    return sim, unstable_mask


def dedupe_near_duplicates(
    records_with_vectors: List[Tuple[PaperRecord, np.ndarray]],
    threshold: float,
) -> Tuple[List[PaperRecord], List[DropRecord]]:
    if not records_with_vectors:
        return [], []

    records = [r for r, _ in records_with_vectors]
    vectors = np.array([v for _, v in records_with_vectors])
    sim_matrix, _unstable_mask = _cosine_sim_matrix(vectors)

    kept: List[PaperRecord] = []
    kept_indices: List[int] = []
    dropped: List[DropRecord] = []

    for i, record in enumerate(records):
        is_near_dup = False
        for j in kept_indices:
            if sim_matrix[i, j] >= threshold:
                is_near_dup = True
                dropped.append(
                    DropRecord(
                        arxiv_id=record.arxiv_id,
                        title=record.title,
                        reason=f"near-duplicate (cosine similarity {sim_matrix[i, j]:.3f} "
                               f">= threshold {threshold})",
                        kept_arxiv_id=records[j].arxiv_id,
                    )
                )
                logger.debug(
                    f"[{record.sector}] dropped '{record.title[:60]}' "
                    f"({record.arxiv_id}): near-duplicate of "
                    f"'{records[j].title[:60]}' ({records[j].arxiv_id}), "
                    f"sim={sim_matrix[i, j]:.3f}"
                )
                break
        if not is_near_dup:
            kept.append(record)
            kept_indices.append(i)

    return kept, dropped