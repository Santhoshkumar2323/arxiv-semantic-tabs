from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from pipeline.fetch import PaperRecord
from pipeline.logger import get_logger

logger = get_logger()

@dataclass
class RankedPaper:
    paper: PaperRecord
    score: float


@dataclass
class ScoreStats:
    min: float
    max: float
    mean: float


def rank_sector(
    profile_vector: np.ndarray,
    records: List[PaperRecord],
    paper_vectors: np.ndarray,
    top_k: int,
    min_score: float = 0.0,
) -> Tuple[List[RankedPaper], Optional[ScoreStats]]:
    if not records:
        return [], None

    if len(records) != paper_vectors.shape[0]:
        raise ValueError(
            f"records/paper_vectors length mismatch: {len(records)} records "
            f"but {paper_vectors.shape[0]} vectors — these must be aligned "
            f"1:1, likely a bug upstream in how vectors were assembled."
        )

    if profile_vector.shape[0] != paper_vectors.shape[1]:
        raise ValueError(
            f"profile_vector dim ({profile_vector.shape[0]}) doesn't match "
            f"paper_vectors dim ({paper_vectors.shape[1]}) — likely encoded "
            f"with two different models."
        )

    scores = paper_vectors @ profile_vector

    non_finite = ~np.isfinite(scores)

    stats: Optional[ScoreStats] = None
    finite_scores = scores[~non_finite]
    if finite_scores.size:
        stats = ScoreStats(
            min=float(finite_scores.min()),
            max=float(finite_scores.max()),
            mean=float(finite_scores.mean()),
        )
        logger.debug(
            f"rank_sector: score stats — min={stats.min:.4f}, "
            f"max={stats.max:.4f}, mean={stats.mean:.4f}"
        )

    if non_finite.any():
        bad_records = [records[i] for i in np.where(non_finite)[0]]
        logger.warning(
            f"rank_sector: {len(bad_records)} paper(s) produced a non-finite "
            f"score (corrupted embedding?) — excluding from ranking rather "
            f"than trusting the value: "
            + ", ".join(f"{r.arxiv_id} ({r.title[:40]})" for r in bad_records)
        )
        scores = np.where(non_finite, -np.inf, scores)

    ranked_indices = np.argsort(-scores)

    results: List[RankedPaper] = []
    for idx in ranked_indices:
        score = float(scores[idx])
        if score < min_score:
            continue
        results.append(RankedPaper(paper=records[idx], score=score))
        if len(results) >= top_k:
            break

    return results, stats