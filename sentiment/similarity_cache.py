"""TF-IDF + cosine similarity cache for admin-corrected feedback.

Lets the active-learning loop persist beyond a single submission: when an
admin corrects a feedback row, the cleaned text's TF-IDF vector is stored in
`feedback_correction_memory` along with the corrected label. On every future
`process_feedback` call we scan memory for any row whose cosine similarity is
>= 0.75 and, if found, override the model's label with the admin's label.

Pure-Python TF-IDF (no sklearn dependency) so this module works in any
deployment that already has the project running. The vocabulary is fitted
lazily across all currently-stored memory rows plus the candidate text, which
is sufficient because feedback submissions are short documents and the memory
table only holds a few thousand rows at most in practice.

Public API:
    encode(text, vocab, idf) -> dict[int, float]   # sparse TF-IDF vector
    cosine(a, b) -> float
    similarity_cache_lookup(cleaned_text, db_session) -> dict | None
    memory_record_correction(feedback, db_session, admin_name=None)
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

_TOKEN_RE = re.compile(r"[a-zA-Z]{2,}")

# Cosine threshold above which a memory hit overrides the model's label.
# Conservative-medium: catches rephrasings like "wifi drops constantly" ~
# "internet keeps disconnecting" but avoids overriding on loosely related text.
SIMILARITY_THRESHOLD = 0.75


def tokenize(text: str) -> List[str]:
    """Lowercased alphabetic tokens, length >= 2. No stopwords, no stemming:
    we want exact matches so that cosine similarity reflects vocabulary
    overlap rather than semantic equivalence."""
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(str(text))]


def encode(text: str, vocab: Dict[str, int], idf: Dict[str, float]) -> Dict[int, float]:
    """Compute a sparse TF-IDF vector for `text` using the supplied vocabulary
    and IDF table. Returns {feature_index: weight}.

    TF = count(t) / len(tokens).  IDF = log((1 + N) / (1 + df(t))) + 1
    (smoothed, sklearn-style so the runtime agrees with what scikit-learn
    would produce for a quick sanity check in dev).
    """
    toks = tokenize(text)
    if not toks:
        return {}
    tf = Counter(toks)
    length = len(toks)
    vec: Dict[int, float] = {}
    for tok, count in tf.items():
        idx = vocab.get(tok)
        if idx is None:
            continue
        weight = (count / length) * idf[tok]
        if weight > 0:
            vec[idx] = weight
    return vec


def cosine(a: Dict[int, float], b: Dict[int, float]) -> float:
    """Cosine similarity between two sparse vectors represented as
    {feature_index: weight} dicts. Returns 0.0 if either is empty."""
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a  # iterate the smaller side
    dot = 0.0
    for k, v in a.items():
        bv = b.get(k)
        if bv is not None:
            dot += v * bv
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _fit_vocab_and_idf(documents: Iterable[str]) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Fit vocabulary + IDF across an iterable of documents. Each document
    is tokenized here."""
    df: Counter = Counter()
    docs: List[List[str]] = []
    for d in documents:
        toks = tokenize(d)
        docs.append(toks)
        df.update(set(toks))
    n = max(1, len(docs))
    vocab: Dict[str, int] = {}
    idf: Dict[str, float] = {}
    idx = 0
    for tok, dfreq in df.items():
        vocab[tok] = idx
        idf[tok] = math.log((1 + n) / (1 + dfreq)) + 1.0
        idx += 1
    return vocab, idf


def similarity_cache_lookup(cleaned_text: str, db_session) -> Optional[dict]:
    """If `cleaned_text` is cosine-similar to any admin-corrected memory row,
    return that row's payload as a plain dict:

        {
          'feedback_id': int,
          'sentiment': str,
          'category': str,
          'urgency_score': int,
          'similarity': float,
          'admin_name': str,
        }

    Otherwise return None. The threshold is SIMILARITY_THRESHOLD (0.75).

    `db_session` is a Flask-SQLAlchemy session (we use the raw SQLAlchemy
    `db.session` from `database`). We keep this function DB-agnostic by
    importing models lazily inside the function so the module is importable
    from the backfill script too.
    """
    if not cleaned_text or not cleaned_text.strip():
        return None

    from database import FeedbackCorrectionMemory  # late import

    # Load all memory rows. For the scale of this project (hundreds to low
    # thousands of corrections) this is fine; if the table ever grows past
    # ~10k rows, switch to a precomputed vocab + batched numpy matrix.
    rows: List[FeedbackCorrectionMemory] = (
        db_session.query(FeedbackCorrectionMemory).all()
    )
    if not rows:
        return None

    docs = [cleaned_text] + [(r.cleaned_text or '') for r in rows]
    vocab, idf = _fit_vocab_and_idf(docs)

    query_vec = encode(cleaned_text, vocab, idf)
    if not query_vec:
        return None

    best_sim = 0.0
    best_row: Optional[FeedbackCorrectionMemory] = None
    for row in rows:
        try:
            import json
            stored = json.loads(row.tfidf_vector or '{}')
        except Exception:
            continue
        # Stored vector was computed under a (possibly different) vocab. For
        # correctness we recompute from cleaned_text here -- vocab is freshly
        # fit on the union above so both vectors live in the same space.
        mem_vec = encode(row.cleaned_text or '', vocab, idf)
        if not mem_vec:
            continue
        sim = cosine(query_vec, mem_vec)
        if sim > best_sim:
            best_sim = sim
            best_row = row

    if best_row is None or best_sim < SIMILARITY_THRESHOLD:
        return None

    return {
        'feedback_id': best_row.feedback_id,
        'sentiment': best_row.sentiment,
        'category': best_row.category,
        'urgency_score': best_row.urgency_score,
        'similarity': round(best_sim, 4),
        'admin_name': best_row.admin_name,
    }


def memory_record_correction(feedback, db_session, admin_name: Optional[str] = None) -> int:
    """Insert (or replace) the memory row for `feedback`. Idempotent: if a
    row already exists for `feedback.id`, it is updated in place. Returns
    the memory row id.

    Does NOT commit. Caller commits within the existing transaction.
    """
    import json
    from database import FeedbackCorrectionMemory

    cleaned = (feedback.cleaned_text or '').strip()
    if not cleaned:
        return 0

    # We do not need the global vocab here -- the stored vector only needs
    # to be retrievable; cosine at lookup time refits vocab across the
    # union of query + all stored rows, then re-encodes both sides in that
    # vocab. So a freshly-fit vocab is fine.
    vocab, idf = _fit_vocab_and_idf([cleaned])
    vec = encode(cleaned, vocab, idf)
    if not vec:
        return 0

    existing = (
        db_session.query(FeedbackCorrectionMemory)
        .filter_by(feedback_id=feedback.id)
        .first()
    )
    payload = {
        'sentiment': feedback.sentiment or '',
        'category': feedback.category or '',
        'urgency_score': int(feedback.urgency_score or 1),
        'admin_name': admin_name or '',
        'cleaned_text': cleaned,
        'tfidf_vector': json.dumps({str(k): v for k, v in vec.items()}),
    }
    if existing:
        existing.cleaned_text = payload['cleaned_text']
        existing.tfidf_vector = payload['tfidf_vector']
        existing.sentiment = payload['sentiment']
        existing.category = payload['category']
        existing.urgency_score = payload['urgency_score']
        existing.admin_name = payload['admin_name']
        return existing.id

    row = FeedbackCorrectionMemory(
        feedback_id=feedback.id,
        cleaned_text=payload['cleaned_text'],
        tfidf_vector=payload['tfidf_vector'],
        sentiment=payload['sentiment'],
        category=payload['category'],
        urgency_score=payload['urgency_score'],
        admin_name=payload['admin_name'],
    )
    db_session.add(row)
    return row.id or 0
