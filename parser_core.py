# parser_core.py
# Lean quote + attribution extraction engine (improved for news + transcripts + web material)

import re
from bisect import bisect_right
from typing import List, Dict, Optional, Tuple

# -----------------------------
# Tunables (single source of truth)
# -----------------------------

CTX_WINDOW = 800
GROUP_CTX_WINDOW = 500

# Carry-author safety: only carry forward within this many chars
MAX_CARRY_DISTANCE = 400

# Extraction defaults (editable first)
DEFAULT_MIN_LEN = 30
DEFAULT_MAX_LEN = 240
DEFAULT_MAX_NEWLINES = 1
DEFAULT_MAX_SENTENCES = 6

# -----------------------------
# Regex: normalization + small helpers
# -----------------------------

_SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")
_WS_SPACES_RE = re.compile(r"[ \t]+")

# IMPORTANT: preserve blank lines; only trim spaces/tabs around a newline
_WS_NEWLINE_TRIM_RE = re.compile(r"[ \t]*\n[ \t]*")

_WS_MULTI_RE = re.compile(r"\s+")
_PUNCT_SPACE_RE = re.compile(r"\s+([,.;:!?])")
_DEDUPE_RE = re.compile(r"[\s\"“”‘’'`]+")


def normalize_ws(s: str) -> str:
    """
    Normalize whitespace while preserving paragraph breaks.
    - Keeps blank lines intact (critical for paragraph-scoped attribution + paragraph quote scanning).
    """
    s = s.replace("\u00a0", " ")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _WS_SPACES_RE.sub(" ", s)
    s = _WS_NEWLINE_TRIM_RE.sub("\n", s)
    return s.strip()


def normalize_key(text: str) -> str:
    t = normalize_ws(text).lower().strip(" \t\r\n\"'“”‘’")
    return _WS_MULTI_RE.sub(" ", t)


def dedupe_key(s: str) -> str:
    s = normalize_ws(s).lower()
    s = _DEDUPE_RE.sub(" ", s)
    return s.strip(" .,!?:;()-")


def build_line_starts(raw: str) -> List[int]:
    starts = [0]
    for m in re.finditer(r"\n", raw):
        starts.append(m.end())
    return starts


def sentence_count_upto(text: str, limit: int) -> int:
    t = text.strip()
    if not t:
        return 0
    count = 0
    for _ in _SENT_SPLIT_RE.finditer(t):
        count += 1
        if count > limit:
            return count
    return max(1, count)


def looks_like_noise(q: str) -> bool:
    if re.search(r"https?://|www\.", q, re.IGNORECASE):
        return True
    digits = sum(ch.isdigit() for ch in q)
    return (len(q) > 0) and (digits / len(q) > 0.30)


def tidy_quote_text(q: str) -> str:
    q = normalize_ws(q)
    q = _PUNCT_SPACE_RE.sub(r"\1", q)
    q = q.strip()
    if q.endswith(","):
        q = q[:-1].rstrip()
    return q


def clamp_minimal(
    text: str,
    min_len: int,
    max_len: int,
    max_newlines: int,
    max_sentences: int,
) -> Optional[str]:
    t = tidy_quote_text(text)
    if not (min_len <= len(t) <= max_len):
        return None
    if t.count("\n") > max_newlines:
        return None
    if sentence_count_upto(t, max_sentences) > max_sentences:
        return None
    if looks_like_noise(t):
        return None
    return t


def fast_context_norm(s: str) -> str:
    # cheaper than normalize_ws(); used only for attribution context windows
    s = s.replace("\u00a0", " ")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", " ")
    s = _WS_SPACES_RE.sub(" ", s)
    return s.strip()


def chunk_quote_to_maxlen(text: str, max_len: int) -> List[str]:
    """
    Split long quotes into sentence-ish chunks <= max_len.
    Keeps recall without increasing max_len.
    """
    t = tidy_quote_text(text)
    if len(t) <= max_len:
        return [t]

    parts: List[str] = []
    start = 0
    for m in re.finditer(r"[.!?]+(?:\s+|$)", t):
        end = m.end()
        sent = t[start:end].strip()
        if sent:
            parts.append(sent)
        start = end
    if start < len(t):
        tail = t[start:].strip()
        if tail:
            parts.append(tail)

    chunks: List[str] = []
    buf = ""
    for sent in parts:
        if not buf:
            buf = sent
        elif len(buf) + 1 + len(sent) <= max_len:
            buf = f"{buf} {sent}"
        else:
            chunks.append(buf)
            buf = sent
    if buf:
        chunks.append(buf)

    final: List[str] = []
    for c in chunks:
        if len(c) <= max_len:
            final.append(c)
            continue
        sub = re.split(r"(?<=[,;:])\s+", c)
        b = ""
        for s in sub:
            if not b:
                b = s
            elif len(b) + 1 + len(s) <= max_len:
                b = f"{b} {s}"
            else:
                final.append(b)
                b = s
        if b:
            final.append(b)

    return [x.strip() for x in final if x.strip()]


# -----------------------------
# Speech-like / scare-quote filtering
# -----------------------------

_SPEECH_HINT_RE = re.compile(
    r"\b(i|we|you|my|our|me|us|i'm|we're|you're|don't|can't|won't|it's|that's|there's)\b",
    re.IGNORECASE,
)

def speech_like_quote(q: str) -> bool:
