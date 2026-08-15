from __future__ import annotations
import argparse
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

LOCAL_QUICK_OUTPUT_DIR = REPO_ROOT / "tests" / "local_output"


def _check_dependencies() -> bool:
    missing = []
    for module_name, pip_name in [
        ("requests", "requests"),
        ("yaml", "PyYAML"),
        ("numpy", "numpy"),
        ("sentence_transformers", "sentence-transformers"),
    ]:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install them first:  pip install -r requirements.txt")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local dry run of the arXiv dashboard pipeline"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke test: caps pulls at 5/sector, top_k at 3, and (unless "
             "--sector is also given) runs only the first sector. Writes to "
             "tests/local_output/, never touches real data/.",
    )
    parser.add_argument(
        "--sector",
        type=str,
        default=None,
        help="Restrict the run to one sector by name (e.g. --sector RL).",
    )
    args = parser.parse_args()

    if not _check_dependencies():
        sys.exit(1)

    import yaml
    from pipeline.run_pipeline import (
        CONFIG_PATH,
        PAPERS_OUTPUT_PATH,
        RUN_LOG_OUTPUT_PATH,
        load_config,
        run,
    )

    config = load_config(CONFIG_PATH)

    if args.sector:
        matched = [s for s in config["sectors"] if s["name"] == args.sector]
        if not matched:
            names = [s["name"] for s in config["sectors"]]
            print(f"No sector named '{args.sector}'. Available: {', '.join(names)}")
            sys.exit(1)
        config["sectors"] = matched

    papers_output_path = PAPERS_OUTPUT_PATH
    run_log_output_path = RUN_LOG_OUTPUT_PATH

    if args.quick:
        config["max_pull_per_sector"] = 5
        for sector_cfg in config["sectors"]:
            sector_cfg["top_k"] = min(sector_cfg.get("top_k", config["default_top_k"]), 3)
        if not args.sector:
            config["sectors"] = config["sectors"][:1]

        LOCAL_QUICK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        papers_output_path = LOCAL_QUICK_OUTPUT_DIR / "papers.json"
        run_log_output_path = LOCAL_QUICK_OUTPUT_DIR / "run_log.json"

        print(
            f"--quick mode: sector(s) {[s['name'] for s in config['sectors']]}, "
            f"max_pull_per_sector=5, top_k<=3"
        )
        print(f"Output isolated to {LOCAL_QUICK_OUTPUT_DIR}/ — real data/ is untouched.\n")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(config, tmp)
        tmp_config_path = Path(tmp.name)

    print(f"Sectors this run: {[s['name'] for s in config['sectors']]}")
    print(
        f"Embedding model: {config['embedding_model']} "
        f"(first run downloads it from Hugging Face — subsequent runs reuse the local cache)"
    )
    print("Starting...\n")

    start = time.monotonic()
    try:
        run(
            config_path=tmp_config_path,
            papers_output_path=papers_output_path,
            run_log_output_path=run_log_output_path,
        )
    finally:
        tmp_config_path.unlink(missing_ok=True)
    elapsed = time.monotonic() - start

    print(f"\nDone in {elapsed:.1f}s.")
    print(f"Output written to:\n  {papers_output_path}\n  {run_log_output_path}")
    print(
        "Check run_log.json's status table above first — it already tells you "
        "if anything failed per sector before you go digging into papers.json."
    )


if __name__ == "__main__":
    main()