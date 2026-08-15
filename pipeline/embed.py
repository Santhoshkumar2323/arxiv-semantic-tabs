from __future__ import annotations
from typing import Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer
from pipeline.fetch import PaperRecord
from pipeline.logger import get_logger

logger = get_logger()

QUERY_PROMPT = "Represent this sentence for searching relevant passages: "

_model_cache: Dict[str, SentenceTransformer] = {}

def load_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        logger.debug(f"Loading embedding model: {model_name}")
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def _batch_token_counts(model: SentenceTransformer, texts: List[str], prompt: str = "") -> List[int]:
    if not texts:
        return []
    processed = model.preprocess(list(texts), prompt=prompt or None)
    attention_mask = processed.get("attention_mask")
    if attention_mask is not None:
        return [int(x) for x in attention_mask.sum(dim=1).tolist()]
    input_ids = processed.get("input_ids")
    if input_ids is not None:
        return [int(input_ids.shape[1])] * len(texts)
    return [0] * len(texts) 


def _log_truncations(model: SentenceTransformer, labels: List[str], token_counts: List[int]) -> None:
    max_len = model.max_seq_length or 512
    for label, count in zip(labels, token_counts):
        if count > max_len:
            logger.warning(
                f"Truncation: '{label}' is {count} tokens, exceeds the "
                f"model's {max_len}-token limit — will be cut off at encode time."
            )


def encode_profile(model: SentenceTransformer, profile_text: str, sector_name: str) -> np.ndarray:
    text = profile_text.strip()
    counts = _batch_token_counts(model, [text], prompt=QUERY_PROMPT)
    _log_truncations(model, [f"[{sector_name}] profile"], counts)
    vector = model.encode_query(text, prompt=QUERY_PROMPT, normalize_embeddings=True)
    return np.asarray(vector)


def _passage_text(record: PaperRecord) -> str:
    return f"{record.title.strip()}. {record.abstract.strip()}"


def encode_papers(model: SentenceTransformer, records: List[PaperRecord]) -> np.ndarray:
    if not records:
        dim = model.get_sentence_embedding_dimension() or 768
        return np.empty((0, dim))

    texts = [_passage_text(r) for r in records]
    labels = [f"[{r.sector}] {r.arxiv_id}: {r.title[:50]}" for r in records]
    counts = _batch_token_counts(model, texts)
    _log_truncations(model, labels, counts)

    vectors = model.encode_document(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors)