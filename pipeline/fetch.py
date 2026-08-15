from __future__ import annotations
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import requests
from pipeline.logger import get_logger

logger = get_logger()

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"

USER_AGENT = "arxiv-signal-dashboard/1.0 (personal research paper aggregator)"
MIN_REQUEST_INTERVAL_SECONDS = 3.0
MAX_RETRIES = 3
RETRY_BACKOFF_BASE_SECONDS = 5.0


@dataclass
class PaperRecord:
    arxiv_id: str             
    arxiv_id_versioned: str   
    title: str
    abstract: str
    authors: List[str]
    published: str
    pdf_url: str
    categories: List[str]
    sector: str = ""         


class _RateLimiter:
    def __init__(self, min_interval: float = MIN_REQUEST_INTERVAL_SECONDS):
        self.min_interval = min_interval
        self._last_call: Optional[float] = None

    def wait(self) -> None:
        if self._last_call is not None:
            elapsed = time.monotonic() - self._last_call
            remaining = self.min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_call = time.monotonic()

_rate_limiter = _RateLimiter()

def _strip_version(id_url: str) -> Tuple[str, str]:
    tail = id_url.rsplit("/", 1)[-1]
    versioned = tail
    base = tail.rsplit("v", 1)[0] if "v" in tail else tail
    return base, versioned


def build_search_query(categories: List[str], cycle_hours: int, buffer_hours: int = 24) -> str:
    cat_clause = " OR ".join(f"cat:{c}" for c in categories)
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=cycle_hours + buffer_hours)
    date_fmt = "%Y%m%d%H%M"
    date_clause = f"submittedDate:[{start.strftime(date_fmt)} TO {now.strftime(date_fmt)}]"
    return f"({cat_clause}) AND {date_clause}"


def _parse_retry_after(response: requests.Response) -> Optional[float]:
    header = response.headers.get("Retry-After")
    if not header:
        return None
    try:
        return float(header)  
    except ValueError:
        return None  


def _request_with_retry(params: dict) -> str:
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        _rate_limiter.wait()
        try:
            response = requests.get(
                ARXIV_API_URL, params=params, timeout=30,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            return response.text
        except requests.HTTPError as exc:
            last_exc = exc
            retry_after = _parse_retry_after(exc.response) if exc.response is not None else None
            wait_s = retry_after if retry_after is not None else RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            logger.debug(
                f"arXiv request failed (attempt {attempt}/{MAX_RETRIES}): {exc}. "
                f"{'Retry-After header: ' + str(wait_s) + 's' if retry_after is not None else 'Retrying in ' + str(wait_s) + 's (backoff).'}"
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait_s)
        except requests.RequestException as exc:
            last_exc = exc
            wait_s = RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            logger.debug(
                f"arXiv request failed (attempt {attempt}/{MAX_RETRIES}): {exc}. "
                f"Retrying in {wait_s}s."
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait_s)
    raise RuntimeError(f"arXiv API failed after {MAX_RETRIES} attempts: {last_exc}")


def parse_entries(xml_text: str) -> List[PaperRecord]:
    root = ET.fromstring(xml_text)
    records: List[PaperRecord] = []

    for entry in root.findall(f"{ATOM_NS}entry"):
        id_url = (entry.findtext(f"{ATOM_NS}id", default="") or "").strip()
        base_id, versioned_id = _strip_version(id_url)
        title = " ".join((entry.findtext(f"{ATOM_NS}title", default="") or "").split())
        abstract = " ".join((entry.findtext(f"{ATOM_NS}summary", default="") or "").split())
        published = (entry.findtext(f"{ATOM_NS}published", default="") or "").strip()

        authors = [
            (author.findtext(f"{ATOM_NS}name", default="") or "").strip()
            for author in entry.findall(f"{ATOM_NS}author")
        ]

        categories = [
            cat.get("term", "")
            for cat in entry.findall(f"{ATOM_NS}category")
            if cat.get("term")
        ]

        pdf_url = ""
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
                break
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{base_id}"

        records.append(
            PaperRecord(
                arxiv_id=base_id,
                arxiv_id_versioned=versioned_id,
                title=title,
                abstract=abstract,
                authors=authors,
                published=published,
                pdf_url=pdf_url,
                categories=categories,
            )
        )

    return records


def fetch_sector(
    sector_cfg: dict,
    cycle_hours: int,
    max_pull: int,
    page_size: int = 50,
    buffer_hours: int = 24,
) -> List[PaperRecord]:
    sector_name = sector_cfg["name"]
    categories = sector_cfg["arxiv_categories"]
    query = build_search_query(categories, cycle_hours, buffer_hours=buffer_hours)

    collected: List[PaperRecord] = []
    start = 0

    while len(collected) < max_pull:
        remaining = max_pull - len(collected)
        page_size_this_call = min(page_size, remaining)
        params = {
            "search_query": query,
            "start": start,
            "max_results": page_size_this_call,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        logger.debug(
            f"[{sector_name}] fetching start={start}, max_results={page_size_this_call}"
        )
        xml_text = _request_with_retry(params)
        page_records = parse_entries(xml_text)

        for record in page_records:
            record.sector = sector_name
        collected.extend(page_records)

        logger.debug(
            f"[{sector_name}] page returned {len(page_records)} entries "
            f"(cumulative {len(collected)}/{max_pull})"
        )

        if len(page_records) < page_size_this_call:
            break  

        start += page_size_this_call

    return collected[:max_pull]