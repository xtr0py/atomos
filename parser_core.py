import re
from bisect import bisect_right
from typing import List, Dict, Optional, Tuple

# -----------------------------
# Normalization + basic filters
# -----------------------------

_SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")
_WS_SPACES_RE = re.compile(r"[ \t]+")
_WS_NEWLINE_TRIM_RE = re.compile(r"\s*\n\s*")
_WS_MULTI_RE = re.compile(r"\s+")
_PUNCT_SPACE_RE = re.compile(r"\s+([,.;:!?])")

# Context window constants (tunable, centralized)
CTX_WINDOW = 800
GROUP_CTX_WINDOW = 500

def normalize_ws(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _WS_SPACES_RE.sub(" ", s)
    s = _WS_NEWLINE_TRIM_RE.sub("\n", s)
    return s.strip()

def normalize_key(text: str) -> str:
    t = normalize_ws(text).lower().strip(" \t\r\n\"'“”‘’")
    return _WS_MULTI_RE.sub(" ", t)

def sentence_count_upto(text: str, limit: int) -> int:
    """
    Count sentences but stop once we exceed `limit`.
    """
    t = text.strip()
    if not t:
        return 0
    count = 1
    for _ in _SENT_SPLIT_RE.finditer(t):
        count += 1
        if count > limit:
            return count
    return count

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
    """
    Lightweight normalization for attribution matching.
    Cheaper than normalize_ws() because it doesn't run multiple regex passes.
    """
    s = s.replace("\u00a0", " ")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", " ")
    s = _WS_SPACES_RE.sub(" ", s)
    return s.strip()

# -----------------------------
# Patterns learned from examples
# -----------------------------

GOODREADS_TAGS_RE = re.compile(r"^\s*tags\s*:\s*(.+?)\s*$", re.IGNORECASE)
LIKES_RE = re.compile(r"^\s*\d+\s+likes\s*$", re.IGNORECASE)
LIKE_WORD_RE = re.compile(r"^\s*Like\s*$", re.IGNORECASE)
NUMBER_HEADER_RE = re.compile(r"^\s*Number\s+\w+\s*:\s*(?:\(.+?\))?\s*$", re.IGNORECASE)
ATTRIBUTION_LINE_RE = re.compile(r"^\s*(?:—|―|-)\s*([^,\n]{2,120})(?:,.*)?\s*$")
SPEAKER_LABEL_RE = re.compile(r"^\s*([A-Z][A-Z0-9_ \-]{1,24})\s*:\s*(.+?)\s*$")
TABBED_OR_SPACED_ROW_RE = re.compile(r"\t+| {2,}")
TIMESTAMP_ONLY_RE = re.compile(r"^\s*\d{1,2}:\d{2}(?:\.\d+)?\s*$")
CHROME_RE = re.compile(
    r"^\s*(share this|loading\.\.\.|tagged|post navigation|leave a comment|reply|open in|sign up|newsletter|"
    r"learn more about your ad choices|visit .*adchoices|email us at|click on a timestamp)\b",
    re.IGNORECASE,
)
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]")

def is_noise_line(line: str) -> bool:
    l = line.strip()
    if not l:
        return True
    if TIMESTAMP_ONLY_RE.match(l):
        return True
    if CHROME_RE.match(l):
        return True
    if EMOJI_RE.search(l):
        return True
    if LIKES_RE.match(l) or LIKE_WORD_RE.match(l):
        return True
    return False

def looks_like_headline(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if any(ch in stripped for ch in "“”\""):
        return True if stripped.isupper() else False
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return False
    upper = sum(1 for c in letters if c.isupper())
    ratio = upper / max(1, len(letters))
    return ratio > 0.75 and len(stripped.split()) >= 2

# -----------------------------
# Tag parsing (single source of truth)
# -----------------------------

def parse_tag_line(tag_line: str) -> List[str]:
    seen = set()
    out: List[str] = []
    for t in (p.strip() for p in tag_line.split(",")):
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out

# -----------------------------
# Quote scanning: nesting-aware
# -----------------------------

def scan_quote_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    curly_stack: List[int] = []
    straight_stack: List[int] = []

    for i, ch in enumerate(text):
        if ch == "“":
            curly_stack.append(i)
        elif ch == '"':
            if straight_stack:
                spans.append((straight_stack.pop(), i))
            else:
                straight_stack.append(i)
        elif ch == "”":
            if curly_stack:
                spans.append((curly_stack.pop(), i))
            elif straight_stack:
                spans.append((straight_stack.pop(), i))

    spans.sort(key=lambda x: (x[0], -x[1]))
    outer: List[Tuple[int, int]] = []
    for s, e in spans:
        if outer and s >= outer[-1][0] and e <= outer[-1][1]:
            continue
        outer.append((s, e))
    outer.sort(key=lambda x: x[0])
    return outer

# -----------------------------
# Author resolution helpers
# -----------------------------

NAME_TOKEN = r"[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+(?:[-'][A-ZÀ-ÖØ-Þa-zà-öø-ÿ]+)?"
FULLNAME_RE = re.compile(rf"\b({NAME_TOKEN})(?:\s+({NAME_TOKEN}))(?:\s+({NAME_TOKEN}))?\b")
HONORIFIC_RE = re.compile(r"^(Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.)\s+", re.IGNORECASE)

ATTR_VERBS = [
    "said", "says", "told", "wrote", "stated", "notes", "argued", "added", "explained",
    "joked", "quipped", "teased", "continued", "recalled", "insisted", "admitted", "warned"
]
ATTR_VERB_RE = "|".join(ATTR_VERBS)

AFTER_ATTR_PATTERNS = [
    re.compile(rf"^\s*[,–—-]?\s*({FULLNAME_RE.pattern})\s+(?:{ATTR_VERB_RE})\b", re.IGNORECASE),
    re.compile(rf"^\s*[,–—-]?\s*({NAME_TOKEN}(?:\s+{NAME_TOKEN}){{0,3}})\s*,[^.]*?\b(?:{ATTR_VERB_RE})\b", re.IGNORECASE),
    re.compile(rf"^\s*[,–—-]?\s*({NAME_TOKEN})\s+(?:{ATTR_VERB_RE})\b", re.IGNORECASE),
    re.compile(r"^\s*[,–—-]?\s*(their statement|the statement)\s+(continued|said)\b", re.IGNORECASE),
]
BEFORE_ATTR_PATTERNS = [
    re.compile(rf"({FULLNAME_RE.pattern})\s+(?:{ATTR_VERB_RE})[^“\"]{{0,80}}[:;,]?\s*$", re.IGNORECASE),
    re.compile(rf"({FULLNAME_RE.pattern})\s+(?:{ATTR_VERB_RE})[^“\"]{{0,120}}[:]\s*$", re.IGNORECASE),
    re.compile(rf"\bplea from\s+({FULLNAME_RE.pattern})\b", re.IGNORECASE),
    re.compile(rf"\bthe plea\s+({FULLNAME_RE.pattern})\s+made\b", re.IGNORECASE),
    re.compile(rf"\b({FULLNAME_RE.pattern})\s+telling\b", re.IGNORECASE),
]

def extract_best_person_name(s: str) -> Optional[str]:
    s = normalize_ws(s)
    honorific = ""
    m_h = HONORIFIC_RE.match(s)
    if m_h:
        honorific = m_h.group(1).strip() + " "
        s = HONORIFIC_RE.sub("", s, count=1)

    matches = list(FULLNAME_RE.finditer(s))
    if matches:
        parts = [p for p in matches[-1].groups() if p]
        return honorific + " ".join(parts)

    m = re.search(rf"\b({NAME_TOKEN})\b", s)
    return (honorific + m.group(1)) if m else None

def build_lastname_map(text: str) -> Dict[str, str]:
    text = normalize_ws(text)
    found: Dict[str, set] = {}
    for m in FULLNAME_RE.finditer(text):
        first, second, third = m.group(1), m.group(2), m.group(3)
        if not second:
            continue
        full = " ".join([p for p in [first, second, third] if p])
        last_name = full.split()[-1]
        found.setdefault(last_name, set()).add(full)

    mapping: Dict[str, str] = {}
    for ln, fulls in found.items():
        if len(fulls) == 1:
            mapping[ln] = next(iter(fulls))
    return mapping

def infer_group_author(context_before: str) -> Optional[str]:
    ctx = fast_context_norm(context_before)[-GROUP_CTX_WINDOW:]
    m = re.search(rf"\b({NAME_TOKEN})\s+and\s+({NAME_TOKEN})\s+({NAME_TOKEN})\b", ctx)
    if m:
        return f"{m.group(1)} and {m.group(2)} {m.group(3)}"
    m2 = re.search(rf"\b({NAME_TOKEN})\s+and\s+({NAME_TOKEN})\b", ctx)
    if m2:
        return f"{m2.group(1)} and {m2.group(2)}"
    return None

def resolve_author_for_quote(
    context_before: str,
    context_after: str,
    default_author: str,
    lastname_map: Optional[Dict[str, str]],
    last_known_author_in_paragraph: Optional[str],
) -> str:
    # lightweight normalization for matching
    before = fast_context_norm(context_before)[-CTX_WINDOW:]
    after = fast_context_norm(context_after)[:CTX_WINDOW]

    if re.search(r"\b(their statement|they said|the statement)\b", after, re.IGNORECASE):
        g = infer_group_author(before)
        if g:
            return g

    for pat in AFTER_ATTR_PATTERNS:
        m = pat.search(after)
        if not m:
            continue
        cand = m.group(0)
        if re.search(r"\b(their statement|the statement)\b", cand, re.IGNORECASE):
            return last_known_author_in_paragraph or default_author
        name = extract_best_person_name(cand)
        if name:
            name = name.split(",")[0].strip()
            if lastname_map and len(name.split()) == 1 and name in lastname_map:
                name = lastname_map[name]
            return name

    for pat in BEFORE_ATTR_PATTERNS:
        m = pat.search(before)
        if not m:
            continue
        name = extract_best_person_name(m.group(0))
        if name:
            name = name.split(",")[0].strip()
            if lastname_map and len(name.split()) == 1 and name in lastname_map:
                name = lastname_map[name]
            return name

    return last_known_author_in_paragraph or default_author

# -----------------------------
# Unquoted quote-list extraction
# -----------------------------

def is_author_line_candidate(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 70:
        return False
    if is_noise_line(line):
        return False
    if looks_like_headline(line):
        return False
    if NUMBER_HEADER_RE.match(line):
        return False
    if GOODREADS_TAGS_RE.match(line):
        return False
    letters = sum(1 for c in line if c.isalpha())
    if letters < 3:
        return False
    if line.lower() in {"summary", "transcript", "tagged"}:
        return False
    if ":" in line:
        return False
    return bool(re.search(rf"\b{NAME_TOKEN}\b", line))

def extract_unquoted_quote_lists(
    text: str,
    *,
    default_author: str,
    min_len: int,
    max_len: int,
    max_newlines: int,
    max_sentences: int,
) -> List[Dict[str, object]]:
    lines = [normalize_ws(l) for l in text.splitlines()]
    results: List[Dict[str, object]] = []
    current_author: Optional[str] = None
    last_entry_index: Optional[int] = None

    def add_entry(qtext: str, author: Optional[str], pos: int) -> None:
        nonlocal last_entry_index
        cleaned = clamp_minimal(qtext, min_len, max_len, max_newlines, max_sentences)
        if not cleaned:
            return
        results.append({"text": cleaned, "author": author or default_author, "tags": [], "_pos": pos})
        last_entry_index = len(results) - 1

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        if is_noise_line(line):
            continue

        m_tags = GOODREADS_TAGS_RE.match(line)
        if m_tags and last_entry_index is not None:
            results[last_entry_index]["tags"] = parse_tag_line(m_tags.group(1))
            continue

        m_attr = ATTRIBUTION_LINE_RE.match(line)
        if m_attr and last_entry_index is not None:
            author = tidy_quote_text(m_attr.group(1))
            results[last_entry_index]["author"] = author
            current_author = author
            continue

        if NUMBER_HEADER_RE.match(line):
            continue

        if is_author_line_candidate(line):
            current_author = tidy_quote_text(line)
            continue

        add_entry(line, current_author, pos=idx)

    return results

# -----------------------------
# TSV / table-row extraction
# -----------------------------

def looks_like_author_field(field: str) -> bool:
    f = field.strip()
    if not f or len(f) > 120:
        return False
    if not re.search(rf"\b{NAME_TOKEN}\b", f):
        return False
    return f.lower() not in {"english", "french", "spanish", "german"}

def extract_table_rows(
    text: str,
    *,
    default_author: str,
    min_len: int,
    max_len: int,
    max_newlines: int,
    max_sentences: int,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for i, line in enumerate(text.splitlines()):
        if is_noise_line(line):
            continue
        if "\t" not in line and not re.search(r" {2,}", line):
            continue

        parts = [p.strip() for p in TABBED_OR_SPACED_ROW_RE.split(line) if p.strip()]
        if len(parts) < 2:
            continue

        quote_field, author_field = parts[0], parts[1]
        if not looks_like_author_field(author_field):
            continue

        cleaned = clamp_minimal(quote_field, min_len, max_len, max_newlines, max_sentences)
        if not cleaned:
            continue

        results.append({"text": cleaned, "author": tidy_quote_text(author_field), "tags": [], "_pos": i})
    return results

# -----------------------------
# Main extract_quotes (all modes)
# -----------------------------

def extract_quotes(
    text: str,
    default_author: str = "Unknown",
    *,
    min_len: int = 30,
    max_len: int = 240,          # DO NOT CHANGE
    max_newlines: int = 1,
    max_sentences: int = 2,
    enable_speaker_labels: bool = True,
    enable_quote_lists: bool = True,
    enable_table_rows: bool = True,
    enable_paragraph_carry: bool = True,
) -> List[Dict[str, object]]:
    raw = normalize_ws(text)

    results: List[Dict[str, object]] = []
    seen = set()

    # Paragraph ranges + fast paragraph lookup
    para_ranges: List[Tuple[int, int]] = []
    idx = 0
    for m in re.finditer(r"\n\s*\n", raw):
        para_ranges.append((idx, m.start()))
        idx = m.end()
    para_ranges.append((idx, len(raw)))
    para_starts = [a for a, _ in para_ranges]

    def paragraph_index_for_pos(pos: int) -> int:
        return max(0, bisect_right(para_starts, pos) - 1)

    last_author_by_para: Dict[int, str] = {}

    # Lazily build lastname_map only if we actually need it
    lastname_map: Optional[Dict[str, str]] = None
    lastname_map_ready = False

    def get_lastname_map() -> Dict[str, str]:
        nonlocal lastname_map, lastname_map_ready
        if not lastname_map_ready:
            lastname_map = build_lastname_map(raw)
            lastname_map_ready = True
        return lastname_map or {}

    # 1) Quoted spans
    spans = scan_quote_spans(raw)
    for (s, e) in spans:
        inside = raw[s + 1 : e]
        cleaned = clamp_minimal(inside, min_len, max_len, max_newlines, max_sentences)
        if not cleaned:
            continue

        key = normalize_key(cleaned)
        if key in seen:
            continue
        seen.add(key)

        before = raw[max(0, s - CTX_WINDOW) : s]
        after = raw[e + 1 : min(len(raw), e + 1 + CTX_WINDOW)]

        para_i = paragraph_index_for_pos(s)
        carry_author = last_author_by_para.get(para_i) if enable_paragraph_carry else None

        # Only build lastname_map if we might use it:
        # if attribution returns a single-token name we can upgrade later.
        # Here we just pass it lazily as None; resolver will only consult if provided.
        # To preserve ability to upgrade last names, we provide it when there are multiple quotes (typical news style).
        lm = get_lastname_map() if len(spans) >= 2 else None

        author = resolve_author_for_quote(
            before, after,
            default_author=default_author,
            lastname_map=lm,
            last_known_author_in_paragraph=carry_author,
        )

        if enable_paragraph_carry and author and author != default_author:
            last_author_by_para[para_i] = author

        results.append({"text": cleaned, "author": author, "tags": [], "_pos": s})

    # 2) Speaker labels
    if enable_speaker_labels:
        for i, line in enumerate(raw.splitlines()):
            if is_noise_line(line):
                continue
            if GOODREADS_TAGS_RE.match(line):
                continue

            m = SPEAKER_LABEL_RE.match(line.strip())
            if not m:
                continue

            speaker = m.group(1).strip()
            utterance = m.group(2).strip()

            cleaned = clamp_minimal(utterance, min_len, max_len, max_newlines, max_sentences)
            if not cleaned:
                continue

            key = normalize_key(cleaned)
            if key in seen:
                continue
            seen.add(key)

            results.append({"text": cleaned, "author": speaker, "tags": [], "_pos": i})

    # 3) Unquoted quote-lists
    if enable_quote_lists:
        for r in extract_unquoted_quote_lists(
            raw,
            default_author=default_author,
            min_len=min_len,
            max_len=max_len,
            max_newlines=max_newlines,
            max_sentences=max_sentences,
        ):
            key = normalize_key(r["text"])
            if key in seen:
                continue
            seen.add(key)
            results.append(r)

    # 4) Table rows
    if enable_table_rows:
        for r in extract_table_rows(
            raw,
            default_author=default_author,
            min_len=min_len,
            max_len=max_len,
            max_newlines=max_newlines,
            max_sentences=max_sentences,
        ):
            key = normalize_key(r["text"])
            if key in seen:
                continue
            seen.add(key)
            results.append(r)

    # Post-pass: attach Goodreads tags/attribution to nearest previous quote
    results.sort(key=lambda x: int(x.get("_pos", 0)))
    positions = [int(r.get("_pos", 0)) for r in results]

    def prev_index(pos: int) -> Optional[int]:
        j = bisect_right(positions, pos) - 1
        return j if j >= 0 else None

    running_pos = 0
    for line in raw.splitlines():
        stripped = line.strip()

        m_tags = GOODREADS_TAGS_RE.match(stripped)
        if m_tags:
            j = prev_index(running_pos)
            if j is not None:
                results[j]["tags"] = parse_tag_line(m_tags.group(1))

        m_attr = ATTRIBUTION_LINE_RE.match(stripped)
        if m_attr:
            j = prev_index(running_pos)
            if j is not None and results[j]["author"] == default_author:
                results[j]["author"] = tidy_quote_text(m_attr.group(1))

        running_pos += len(line) + 1

    return [{"text": r["text"], "author": r["author"], "tags": r.get("tags", [])} for r in results]
