from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
import yaml

from pipeline.dedupe import dedupe_by_id, dedupe_near_duplicates
from pipeline.embed import encode_papers, encode_profile, load_model
from pipeline.fetch import fetch_sector
from pipeline.logger import RunTracker, get_logger
from pipeline.rank import rank_sector

logger = get_logger()

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "sectors.yaml"
PAPERS_OUTPUT_PATH = REPO_ROOT / "data" / "papers.json"
RUN_LOG_OUTPUT_PATH = REPO_ROOT / "data" / "run_log.json"


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

SUPPORTED_ZERO_PAPER_POLICIES = {"strict_empty"}


def _validate_zero_paper_policy(config: Dict[str, Any]) -> None:
    policy = config.get("zero_paper_policy", "strict_empty")
    if policy not in SUPPORTED_ZERO_PAPER_POLICIES:
        raise NotImplementedError(
            f"zero_paper_policy: '{policy}' is set in config but not implemented. "
            f"Only {SUPPORTED_ZERO_PAPER_POLICIES} is currently supported."
        )



def _empty_sector_result(pulled: int = 0) -> Dict[str, Any]:
    return {"pulled": pulled, "retained": 0, "papers": []}


def _process_sector(
    sector_cfg: Dict[str, Any],
    config: Dict[str, Any],
    model,
    status,
) -> Dict[str, Any]:
    sector_name = sector_cfg["name"]

    raw_records = fetch_sector(
        sector_cfg,
        cycle_hours=config["cycle_hours"],
        max_pull=config.get("max_pull_per_sector", 30),
        buffer_hours=config.get("window_buffer_hours", 24),
    )
    status.pulled = len(raw_records)
    if not raw_records:
        return _empty_sector_result(pulled=0)

    id_kept, id_dropped = dedupe_by_id(raw_records)
    logger.debug(f"[{sector_name}] exact-ID dedup: dropped {len(id_dropped)}")
    if not id_kept:
        return _empty_sector_result(pulled=len(raw_records))

    paper_vectors = encode_papers(model, id_kept)
    vector_by_id = {r.arxiv_id: v for r, v in zip(id_kept, paper_vectors)}

    near_dup_kept, near_dup_dropped = dedupe_near_duplicates(
        list(zip(id_kept, paper_vectors)),
        threshold=config.get("near_duplicate_threshold", 0.92),
    )
    logger.debug(f"[{sector_name}] near-duplicate dedup: dropped {len(near_dup_dropped)}")
    if not near_dup_kept:
        return _empty_sector_result(pulled=len(raw_records))
    final_vectors = np.array([vector_by_id[r.arxiv_id] for r in near_dup_kept])

    profile_vector = encode_profile(model, sector_cfg["profile"], sector_name)
    top_k = sector_cfg.get("top_k", config.get("default_top_k", 10))
    ranked = rank_sector(
        profile_vector,
        near_dup_kept,
        final_vectors,
        top_k=top_k,
        min_score=config.get("relevance_min_score", 0.0),
    )
    status.retained = len(ranked)

    papers_out = [
        {
            "arxiv_id": rp.paper.arxiv_id,
            "title": rp.paper.title,
            "abstract": rp.paper.abstract,
            "authors": rp.paper.authors,
            "published": rp.paper.published,
            "pdf_url": rp.paper.pdf_url,
            "score": round(rp.score, 4),
        }
        for rp in ranked
    ]
    return {"pulled": len(raw_records), "retained": len(ranked), "papers": papers_out}


def run(
    config_path: Path = CONFIG_PATH,
    papers_output_path: Path = None,
    run_log_output_path: Path = None,
) -> None:
    papers_output_path = papers_output_path or PAPERS_OUTPUT_PATH
    run_log_output_path = run_log_output_path or RUN_LOG_OUTPUT_PATH

    config = load_config(config_path)
    _validate_zero_paper_policy(config)

    try:
        model = load_model(config["embedding_model"])
    except Exception as exc:
        logger.error(f"FATAL: could not load embedding model '{config.get('embedding_model')}': {exc}")
        raise

    tracker = RunTracker(cycle_hours=config["cycle_hours"])
    papers_output: Dict[str, Any] = {}

    for sector_cfg in config["sectors"]:
        sector_name = sector_cfg["name"]
        with tracker.track_sector(sector_name) as status:
            papers_output[sector_name] = _process_sector(sector_cfg, config, model, status)

        if sector_name not in papers_output:
            papers_output[sector_name] = _empty_sector_result(pulled=status.pulled)

    papers_output_path.parent.mkdir(parents=True, exist_ok=True)
    papers_output_path.write_text(json.dumps(papers_output, indent=2))
    logger.info(f"Wrote {papers_output_path}")

    tracker.finalize(run_log_path=run_log_output_path)


if __name__ == "__main__":
    run()