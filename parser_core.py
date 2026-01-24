# parser_core.py
# Lean quote + attribution extraction engine (improved for news + transcripts + web material)

import re
from bisect import bisect_right
from typing import List, Dict, Optional, Tuple

# -----------------------------
# Tunables
# -----------------------------

CTX_WINDOW = 800
GROUP_CTX_WINDOW = 500

# -----------------------------
# Regex: normalization + small helpers
# -----------------------------

_SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")
_WS_SPACES_RE = re.compile(r"[ \t]+")
_WS_NEWLINE_TRIM_RE = re.compile(r"\s*\n\s*")
_WS_MULTI_RE = re.compile(r"\s+")
_PUNCT_SPACE_RE = re.compile(r"\s+([,.;:!?])")
_DEDUPE_RE = re.compile(r"[\s\"“”‘’'`]+")

def normalize_ws(s: str) -> str:
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
# Regex: noise + structural markers
# -----------------------------

GOODREADS_TAGS_RE = re.compile(r"^\s*tags\s*:\s*(.+?)\s*$", re.IGNORECASE)
LIKES_RE = re.compile(r"^\s*\d+\s+likes\s*$", re.IGNORECASE)
LIKE_WORD_RE = re.compile(r"^\s*Like\s*$", re.IGNORECASE)
NUMBER_HEADER_RE = re.compile(r"^\s*Number\s+\w+\s*:\s*(?:\(.+?\))?\s*$", re.IGNORECASE)
ATTRIBUTION_LINE_RE = re.compile(r"^\s*(?:—|―|-)\s*([^,\n]{2,120})(?:,.*)?\s*$")
SPEAKER_LABEL_RE = re.compile(r"^\s*([A-Z][A-Z0-9_ \-]{1,24})\s*:\s*(.+?)\s*$")

# transcripts often use Title Case speaker labels (Host:, Fortney:, Kristen Fortney:)
TITLE_SPEAKER_LABEL_RE = re.compile(r"^\s*([A-Z][A-Za-z.'\- ]{1,40})\s*:\s*(.+?)\s*$")
BAD_SPEAKER_LABELS = {
    "Transcript", "Advertisement", "Sponsored", "Related", "Read More", "More", "Note",
    "Sign Up", "Newsletter", "Latest", "Breaking", "Update",
}

TABBED_OR_SPACED_ROW_RE = re.compile(r"\t+| {2,}")
TIMESTAMP_ONLY_RE = re.compile(r"^\s*\d{1,2}:\d{2}(?:\.\d+)?\s*$")
CHROME_RE = re.compile(
    r"^\s*(share this|loading\.\.\.|tagged|post navigation|leave a comment|reply|open in|sign up|newsletter|"
    r"keep up with|keep up to date|learn more about your ad choices|visit .*adchoices|email us at|"
    r"click on a timestamp|read more|read the full story|more from|related (articles|stories)|recommended|"
    r"cookie (policy|preferences)|privacy policy|terms of service|all rights reserved)\b",
    re.IGNORECASE,
)
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]")

# Quote-collection cues (to avoid harvesting normal prose)
BULLET_RE = re.compile(r"^\s*(?:[-*•‣▪]|(\d+)[.)])\s+")
QUOTEY_LINE_RE = re.compile(r"[\"“”]")

def looks_like_nav_line(line: str) -> bool:
    l = line.strip()
    if len(l) < 8:
        return False
    seps = sum(l.count(x) for x in ["|", "•", "»", "›", "—"])
    return seps >= 3

def is_noise_line(line: str) -> bool:
    l = line.strip()
    if not l:
        return True
    if TIMESTAMP_ONLY_RE.match(l):
        return True
    if CHROME_RE.match(l):
        return True
    if looks_like_nav_line(l):
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
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return False
    upper = sum(1 for c in letters if c.isupper())
    ratio = upper / max(1, len(letters))
    return ratio > 0.75 and len(stripped.split()) >= 2

def has_quote_collection_cues(text: str) -> bool:
    nonempty = [l.rstrip() for l in text.splitlines() if l.strip()]
    if len(nonempty) < 2:
        return False

    bulletish = sum(1 for l in nonempty if BULLET_RE.match(l))
    attributions = sum(1 for l in nonempty if ATTRIBUTION_LINE_RE.match(l.strip()))
    tags = sum(1 for l in nonempty if GOODREADS_TAGS_RE.match(l.strip()))
    quoted = sum(1 for l in nonempty if QUOTEY_LINE_RE.search(l))

    authorish = sum(1 for l in nonempty if is_author_line_candidate(l))

    if tags > 0 or attributions > 0:
        return True
    if bulletish >= 2:
        return True
    if quoted >= 2 and len(nonempty) <= 30:
        return True
    if authorish >= 2 and len(nonempty) <= 60:
        return True
    return False

# -----------------------------
# Tag parsing
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
# Quoted-span scanning (paragraph-scoped, nesting-aware, resilient)
# -----------------------------

_PARA_SPLIT_RE = re.compile(r"\n\s*\n")

def scan_quote_spans(text: str) -> List[Tuple[int, int]]:
    """
    Returns outermost spans for “...” and "..." and «...» (best-effort).

    Improvements:
      - scans per paragraph to prevent cross-paragraph poisoning
      - ignores likely dangling closing straight quotes when stack is empty
      - ignores OCR/apostrophe-like quotes inside words (I"m, don"t)
      - still avoids inch marks (5")
    """
    spans: List[Tuple[int, int]] = []
    for p_start, p_end in _paragraph_ranges(text):
        spans.extend(_scan_quote_spans_in_block(text, p_start, p_end))

    spans.sort(key=lambda x: (x[0], x[1]))
    return _outermost_intervals(spans)

def _paragraph_ranges(text: str) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    start = 0
    for m in _PARA_SPLIT_RE.finditer(text):
        end = m.start()
        if start < end:
            ranges.append((start, end))
        start = m.end()
    if start < len(text):
        ranges.append((start, len(text)))
    return ranges

def _scan_quote_spans_in_block(text: str, start: int, end: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    curly_stack: List[int] = []
    straight_stack: List[int] = []
    angle_stack: List[int] = []

    def is_escaped(i: int) -> bool:
        return i > 0 and text[i - 1] == "\\"

    def looks_like_inch_mark(i: int) -> bool:
        return i > 0 and text[i - 1].isdigit()

    def is_word_char(ch: str) -> bool:
        return ch.isalpha() or ch.isdigit() or ch == "_"

    def looks_like_apostrophe_quote(i: int) -> bool:
        # I"m, don"t, we"re (OCR/mangled apostrophes)
        if i <= 0 or i + 1 >= len(text):
            return False
        return is_word_char(text[i - 1]) and is_word_char(text[i + 1])

    def looks_like_dangling_closer(i: int) -> bool:
        # ... respect", after ...
        if i <= 0:
            return False
        if not is_word_char(text[i - 1]):
            return False
        j = i + 1
        while j < len(text) and text[j].isspace():
            j += 1
        if j >= len(text):
            return False
        return text[j] in ",.;:)]}"

    for i in range(start, end):
        ch = text[i]

        if ch == "“":
            curly_stack.append(i)
            continue
        if ch == "”":
            if curly_stack:
                out.append((curly_stack.pop(), i))
            continue

        if ch == "«":
            angle_stack.append(i)
            continue
        if ch == "»":
            if angle_stack:
                out.append((angle_stack.pop(), i))
            continue

        if ch != '"':
            continue

        if is_escaped(i) or looks_like_inch_mark(i) or looks_like_apostrophe_quote(i):
            continue

        if straight_stack:
            out.append((straight_stack.pop(), i))
            continue

        if looks_like_dangling_closer(i):
            continue

        straight_stack.append(i)

    return out

def _outermost_intervals(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    # start asc, end desc -> outer spans first
    spans = sorted(spans, key=lambda x: (x[0], -x[1]))
    outer: List[Tuple[int, int]] = []
    cur_start = -1
    cur_end = -1
    for s, e in spans:
        if not outer:
            outer.append((s, e))
            cur_start, cur_end = s, e
            continue
        if s >= cur_start and e <= cur_end:
            continue
        outer.append((s, e))
        cur_start, cur_end = s, e
    outer.sort(key=lambda x: x[0])
    return outer

# -----------------------------
# Attribution (names + verbs + orgs)
# -----------------------------

NAME_WORD = r"[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+(?:[-'][A-ZÀ-ÖØ-Þa-zà-öø-ÿ]+)?"
INITIALS  = r"(?:[A-Z]\.){1,3}"  # J. or J.K. or J. K.
PARTICLE  = r"(?:de|del|da|di|la|le|van|von|der|den|du|st)\.?"
SUFFIX    = r"(?:Jr\.|Sr\.|II|III|IV)"

NAME_TOKEN = rf"(?:{NAME_WORD}|{INITIALS})"
NAME_PHRASE = rf"{NAME_TOKEN}(?:\s+(?:{PARTICLE}\s+)?{NAME_TOKEN}){{0,4}}(?:\s+{SUFFIX})?"

FULLNAME_RE = re.compile(
    rf"\b({NAME_TOKEN})(?:\s+(?:{PARTICLE}\s+)?({NAME_TOKEN}))?"
    rf"(?:\s+(?:{PARTICLE}\s+)?({NAME_TOKEN}))?(?:\s+(?:{PARTICLE}\s+)?({NAME_TOKEN}))?"
    rf"(?:\s+({SUFFIX}))?\b"
)
HONORIFIC_RE = re.compile(r"^(Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.)\s+", re.IGNORECASE)

ATTR_VERB_RE = (
    r"(?:according to|said|says|told|wrote|stated|notes|argued|added|explained|"
    r"joked|quipped|teased|continued|recalled|insisted|admitted|warned|"
    r"posted|tweeted|said in a statement|wrote in a statement|told reporters)"
)

ROLE_PREFIX = r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}\s+)?"

PRONOUN_AFTER_RE = re.compile(
    rf"^\s*[,–—-]?\s*(he|she|they)\s+(?:{ATTR_VERB_RE})\b",
    re.IGNORECASE,
)

ORG_SUFFIX = (
    r"(?:Association|Society|College|Academy|Committee|Center|Centre|Agency|Department|Ministry|Office|"
    r"Institute|Institution|University|Hospital|Clinic|Council|Board|Commission|Organization|Organisation|"
    r"Foundation|Federation|Alliance|Group|Administration|CDC|WHO|AMA|NIH|FDA|UN|U\.N\.)"
)

ORG_PHRASE_RE = re.compile(
    rf"\b([A-Z][A-Za-z&.\-']+(?:\s+[A-Z][A-Za-z&.\-']+){{0,8}}\s+{ORG_SUFFIX})\b"
)

GENERIC_ORG_WORDS = {"the group", "the agency", "the department", "the committee", "the organization", "the organisation"}

AFTER_ORG_ATTR_PATTERNS = [
    re.compile(rf"^\s*[,–—-]?\s*(the group|the agency|the department|the committee|the organization|the organisation)\s+(?:{ATTR_VERB_RE})\b", re.IGNORECASE),
    re.compile(rf"^\s*[,–—-]?\s*({ORG_PHRASE_RE.pattern})\s+(?:{ATTR_VERB_RE})\b", re.IGNORECASE),
]

BEFORE_ORG_ATTR_PATTERNS = [
    re.compile(rf"({ORG_PHRASE_RE.pattern})\s+(?:{ATTR_VERB_RE}).{{0,160}}[:;,]?\s*$", re.IGNORECASE),
    re.compile(rf"\b(the group|the agency|the department|the committee|the organization|the organisation)\s+(?:{ATTR_VERB_RE}).{{0,160}}[:;,]?\s*$", re.IGNORECASE),
]

AFTER_ATTR_PATTERNS = [
    re.compile(rf"^\s*[,–—-]?\s*({NAME_PHRASE})\s+(?:{ATTR_VERB_RE})\b", re.IGNORECASE),
    re.compile(rf"^\s*[,–—-]?\s*(?:{ATTR_VERB_RE})\s+({NAME_PHRASE})\b", re.IGNORECASE),
    re.compile(rf"^\s*[,–—-]?\s*({ROLE_PREFIX}{NAME_PHRASE})\s+(?:{ATTR_VERB_RE})\b", re.IGNORECASE),
    re.compile(rf"^\s*[,–—-]?\s*({NAME_PHRASE})\s*,[^.]*?\b(?:{ATTR_VERB_RE})\b", re.IGNORECASE),
    re.compile(rf"^\s*[,–—-]?\s*[—–-]\s*({NAME_PHRASE})\b", re.IGNORECASE),
    re.compile(rf"^\s*[,–—-]?\s*[—–-]\s*({ORG_PHRASE_RE.pattern})\b", re.IGNORECASE),
    re.compile(r"^\s*[,–—-]?\s*(their statement|they said|the statement)\s+(continued|said)\b", re.IGNORECASE),
]

BEFORE_ATTR_PATTERNS = [
    re.compile(rf"({NAME_PHRASE})\s+(?:{ATTR_VERB_RE})[^“\"]{{0,80}}[:;,]?\s*$", re.IGNORECASE),
    re.compile(rf"({NAME_PHRASE})\s+(?:{ATTR_VERB_RE})[^“\"]{{0,120}}[:]\s*$", re.IGNORECASE),
    re.compile(rf"\bplea from\s+({NAME_PHRASE})\b", re.IGNORECASE),
    re.compile(rf"\bthe plea\s+({NAME_PHRASE})\s+made\b", re.IGNORECASE),
    re.compile(rf"\b({NAME_PHRASE})\s+telling\b", re.IGNORECASE),
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
        parts = [p for p in parts if p and p.lower().strip(".") not in {"de","del","da","di","la","le","van","von","der","den","du","st"}]
        return honorific + " ".join(parts).strip()

    m = re.search(rf"\b({NAME_TOKEN})\b", s)
    return (honorific + m.group(1)) if m else None

def build_lastname_map(text: str) -> Dict[str, str]:
    text = normalize_ws(text)
    found: Dict[str, set] = {}
    for m in FULLNAME_RE.finditer(text):
        parts = [p for p in m.groups() if p]
        if len(parts) < 2:
            continue
        full = " ".join(parts)
        last = full.split()[-1].strip(".")
        found.setdefault(last, set()).add(full)
    return {ln: next(iter(fulls)) for ln, fulls in found.items() if len(fulls) == 1}

def infer_group_author(context_before: str) -> Optional[str]:
    ctx = fast_context_norm(context_before)[-GROUP_CTX_WINDOW:]
    m = re.search(rf"\b({NAME_TOKEN})\s+and\s+({NAME_TOKEN})\s+({NAME_TOKEN})\b", ctx)
    if m:
        return f"{m.group(1)} and {m.group(2)} {m.group(3)}"
    m2 = re.search(rf"\b({NAME_TOKEN})\s+and\s+({NAME_TOKEN})\b", ctx)
    if m2:
        return f"{m2.group(1)} and {m2.group(2)}"
    return None

def infer_nearest_name_in_before(before: str) -> Optional[str]:
    b = normalize_ws(before)
    matches = list(FULLNAME_RE.finditer(b))
    if matches:
        parts = [p for p in matches[-1].groups() if p]
        parts = [p for p in parts if p and p.lower().strip(".") not in {"de","del","da","di","la","le","van","von","der","den","du","st"}]
        cand = " ".join(parts).strip()
        return cand or None
    m = re.search(rf"\b({NAME_TOKEN})\b", b)
    return m.group(1) if m else None

def infer_nearest_org_in_before(before: str) -> Optional[str]:
    b = normalize_ws(before)
    ms = list(ORG_PHRASE_RE.finditer(b))
    if ms:
        return ms[-1].group(1).strip()
    return None

def _looks_like_quoted_title(before: str, quote: str) -> bool:
    q = quote.strip()
    if len(q) > 90:
        return False
    if not q.endswith("?"):
        return False
    # e.g., podcast, "Why Should I Trust You?"
    return bool(re.search(r"\b(podcast|episode|show|series)\b", before, re.IGNORECASE))

def resolve_author_for_quote(
    context_before: str,
    context_after: str,
    default_author: str,
    lastname_map: Optional[Dict[str, str]],
    last_known_author_in_paragraph: Optional[str],
) -> Tuple[str, str]:
    """
    Returns (author, source) where source is a lightweight confidence hint.
    """
    before = fast_context_norm(context_before)[-CTX_WINDOW:]
    after = fast_context_norm(context_after)[:CTX_WINDOW]

    # Group continuity ("their statement")
    if re.search(r"\b(their statement|they said|the statement)\b", after, re.IGNORECASE):
        g = infer_group_author(before)
        if g:
            return g, "group_infer"

    # Pronoun attribution after quote: “…,” he said.
    if PRONOUN_AFTER_RE.search(after):
        inferred = infer_nearest_name_in_before(before)
        if inferred:
            if lastname_map and len(inferred.split()) == 1 and inferred in lastname_map:
                inferred = lastname_map[inferred]
            return inferred, "after_pronoun"
        # fall through

    # Organization attribution after quote: “…,” the group said / The AMA said.
    for pat in AFTER_ORG_ATTR_PATTERNS:
        m = pat.search(after)
        if not m:
            continue
        src = m.group(1).strip()
        if src.lower() in GENERIC_ORG_WORDS:
            org = infer_nearest_org_in_before(before)
            if org:
                return org, "after_org_infer"
            return (last_known_author_in_paragraph or default_author), "carry_or_default"
        return src, "after_org"

    # Person attribution after quote
    for pat in AFTER_ATTR_PATTERNS:
        m = pat.search(after)
        if not m:
            continue
        if not (m.lastindex and m.lastindex >= 1):
            return (last_known_author_in_paragraph or default_author), "carry_or_default"

        name_src = m.group(1)
        name = extract_best_person_name(name_src)
        if name:
            name = name.split(",")[0].strip()
            if lastname_map and len(name.split()) == 1 and name in lastname_map:
                name = lastname_map[name]
            return name, "after_name"

    # Organization attribution before quote
    for pat in BEFORE_ORG_ATTR_PATTERNS:
        m = pat.search(before)
        if not m:
            continue
        src = m.group(1).strip()
        if src.lower() in GENERIC_ORG_WORDS:
            org = infer_nearest_org_in_before(before)
            if org:
                return org, "before_org_infer"
            return (last_known_author_in_paragraph or default_author), "carry_or_default"
        return src, "before_org"

    # Person attribution before quote
    for pat in BEFORE_ATTR_PATTERNS:
        m = pat.search(before)
        if not m:
            continue
        name_src = m.group(1)
        name = extract_best_person_name(name_src)
        if name:
            name = name.split(",")[0].strip()
            if lastname_map and len(name.split()) == 1 and name in lastname_map:
                name = lastname_map[name]
            return name, "before_name"

    return (last_known_author_in_paragraph or default_author), "carry_or_default"

# -----------------------------
# Quote collections (curated lists)
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

def extract_quote_collections(
    text: str,
    *,
    default_author: str,
    min_len: int,
    max_len: int,
    max_newlines: int,
    max_sentences: int,
    line_starts: List[int],
) -> List[Dict[str, object]]:
    raw_lines = text.splitlines()
    lines = [normalize_ws(l) for l in raw_lines]

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
        if not line or is_noise_line(line):
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

        raw_src = raw_lines[idx]
        is_bullet = bool(BULLET_RE.match(raw_src))
        has_quote_marks = bool(QUOTEY_LINE_RE.search(raw_src))
        if not is_bullet and not has_quote_marks:
            continue

        line_for_quote = BULLET_RE.sub("", raw_src).strip() if is_bullet else line
        add_entry(line_for_quote, current_author, pos=line_starts[idx])

    return results

# -----------------------------
# Tables / TSV rows
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
    line_starts: List[int],
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

        results.append({"text": cleaned, "author": tidy_quote_text(author_field), "tags": [], "_pos": line_starts[i]})
    return results

# -----------------------------
# Inline quoted line + attribution on same line
# -----------------------------

INLINE_QUOTE_ATTR_RE = re.compile(
    r'^\s*[“"](?P<q>.+?)[”"]\s*(?:[—–-]\s*(?P<a>[^,\n]{2,120})|[(\[]\s*(?P<a2>[^)\]\n]{2,120})\s*[)\]])\s*$'
)

def extract_inline_quote_attribution_lines(
    raw: str,
    *,
    default_author: str,
    min_len: int,
    max_len: int,
    max_newlines: int,
    max_sentences: int,
    line_starts: List[int],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for i, line in enumerate(raw.splitlines()):
        if is_noise_line(line):
            continue
        if GOODREADS_TAGS_RE.match(line.strip()):
            continue
        m = INLINE_QUOTE_ATTR_RE.match(line)
        if not m:
            continue

        q_raw = m.group("q")
        candidates = chunk_quote_to_maxlen(q_raw, max_len=max_len) if len(tidy_quote_text(q_raw)) > max_len else [q_raw]
        author = tidy_quote_text(m.group("a") or m.group("a2") or default_author)

        for c in candidates:
            qt = clamp_minimal(c, min_len, max_len, max_newlines, max_sentences)
            if not qt:
                continue
            out.append({"text": qt, "author": author, "tags": [], "_pos": line_starts[i]})
    return out

# -----------------------------
# Main extraction
# -----------------------------

def ok_title_speaker(label: str) -> bool:
    l = label.strip()
    if not l or l in BAD_SPEAKER_LABELS:
        return False
    if len(l) > 40:
        return False
    if len(l.split()) > 6:
        return False
    if l.lower() in {"here’s what to know", "individual autonomy"}:
        return False
    return True

def extract_quotes(
    text: str,
    default_author: str = "Unknown",
    *,
    min_len: int = 30,
    max_len: int = 240,          # default; can be changed by caller
    max_newlines: int = 1,
    max_sentences: int = 6,
    # Modes
    enable_inline_attribution: bool = True,
    enable_quoted_spans: bool = True,
    enable_dialogue_lines: bool = True,
    enable_quote_collections: bool = True,
    enable_tables: bool = True,
    enable_paragraph_attribution: bool = True,
) -> List[Dict[str, object]]:
    """
    Modes:
      - enable_inline_attribution: one-line quote + attribution (e.g., “...”— Author)
      - enable_quoted_spans: quoted spans with contextual attribution
      - enable_dialogue_lines: LABEL: utterance (ALLCAPS and Title Case)
      - enable_quote_collections: curated lists (Goodreads/bullets/quote pages), gated by cues
      - enable_tables: TSV / column rows
      - enable_paragraph_attribution: carry last author within paragraph (gated by confidence)
    """
    raw = normalize_ws(text)
    line_starts = build_line_starts(raw)

    results: List[Dict[str, object]] = []
    seen: set[str] = set()

    # Paragraph index lookup for author carry
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

    lastname_map: Optional[Dict[str, str]] = None
    lastname_map_ready = False

    def get_lastname_map() -> Dict[str, str]:
        nonlocal lastname_map, lastname_map_ready
        if not lastname_map_ready:
            lastname_map = build_lastname_map(raw)
            lastname_map_ready = True
        return lastname_map or {}

    # 1) Inline quote + attribution on the same line
    if enable_inline_attribution:
        for r in extract_inline_quote_attribution_lines(
            raw,
            default_author=default_author,
            min_len=min_len,
            max_len=max_len,
            max_newlines=max_newlines,
            max_sentences=max_sentences,
            line_starts=line_starts,
        ):
            k = dedupe_key(r["text"])
            if k in seen:
                continue
            seen.add(k)
            results.append(r)

    # 2) Quoted spans (contextual attribution)
    spans = scan_quote_spans(raw) if enable_quoted_spans else []
    if spans:
        lm = get_lastname_map() if len(spans) >= 2 else None
        for (s, e) in spans:
            inside = raw[s + 1 : e]
            if not inside.strip():
                continue

            before = raw[max(0, s - CTX_WINDOW) : s]
            after = raw[e + 1 : min(len(raw), e + 1 + CTX_WINDOW)]

            # Optional: drop likely quoted titles (helps if min_len is lowered)
            if _looks_like_quoted_title(before, inside):
                continue

            candidates = chunk_quote_to_maxlen(inside, max_len=max_len) if len(tidy_quote_text(inside)) > max_len else [inside]

            para_i = paragraph_index_for_pos(s)
            carry_author = last_author_by_para.get(para_i) if enable_paragraph_attribution else None

            author, author_src = resolve_author_for_quote(
                before,
                after,
                default_author=default_author,
                lastname_map=lm,
                last_known_author_in_paragraph=carry_author,
            )

            # Only carry forward when attribution is relatively strong
            if enable_paragraph_attribution and author and author != default_author:
                if author_src in {"after_name", "before_name", "after_org", "before_org"}:
                    last_author_by_para[para_i] = author

            for cand in candidates:
                cleaned = clamp_minimal(cand, min_len, max_len, max_newlines, max_sentences)
                if not cleaned:
                    continue

                k = dedupe_key(cleaned)
                if k in seen:
                    continue
                seen.add(k)

                results.append({"text": cleaned, "author": author, "tags": [], "_pos": s})

    # 3) Dialogue lines (LABEL: text)
    if enable_dialogue_lines:
        for i, line in enumerate(raw.splitlines()):
            if is_noise_line(line):
                continue
            if GOODREADS_TAGS_RE.match(line):
                continue

            m = SPEAKER_LABEL_RE.match(line.strip())
            if m:
                speaker = m.group(1).strip()
                utterance = m.group(2).strip()
            else:
                m2 = TITLE_SPEAKER_LABEL_RE.match(line.strip())
                if not m2:
                    continue
                speaker = m2.group(1).strip()
                utterance = m2.group(2).strip()
                if not ok_title_speaker(speaker):
                    continue

            cleaned = clamp_minimal(utterance, min_len, max_len, max_newlines, max_sentences)
            if not cleaned:
                continue

            k = dedupe_key(cleaned)
            if k in seen:
                continue
            seen.add(k)

            results.append({"text": cleaned, "author": speaker, "tags": [], "_pos": line_starts[i]})

    # 4) Quote collections (curated lists) — gated
    if enable_quote_collections and has_quote_collection_cues(raw):
        for r in extract_quote_collections(
            raw,
            default_author=default_author,
            min_len=min_len,
            max_len=max_len,
            max_newlines=max_newlines,
            max_sentences=max_sentences,
            line_starts=line_starts,
        ):
            k = dedupe_key(r["text"])
            if k in seen:
                continue
            seen.add(k)
            results.append(r)

    # 5) Tables / TSV rows
    if enable_tables:
        for r in extract_table_rows(
            raw,
            default_author=default_author,
            min_len=min_len,
            max_len=max_len,
            max_newlines=max_newlines,
            max_sentences=max_sentences,
            line_starts=line_starts,
        ):
            k = dedupe_key(r["text"])
            if k in seen:
                continue
            seen.add(k)
            results.append(r)

    # Post-pass: attach Goodreads tags / attribution lines to nearest previous quote
    results.sort(key=lambda x: int(x.get("_pos", 0)))
    positions = [int(r.get("_pos", 0)) for r in results]

    def prev_index(pos: int) -> Optional[int]:
        j = bisect_right(positions, pos) - 1
        return j if j >= 0 else None

    for i, line in enumerate(raw.splitlines()):
        pos = line_starts[i]
        stripped = line.strip()

        m_tags = GOODREADS_TAGS_RE.match(stripped)
        if m_tags:
            j = prev_index(pos)
            if j is not None:
                results[j]["tags"] = parse_tag_line(m_tags.group(1))

        m_attr = ATTRIBUTION_LINE_RE.match(stripped)
        if m_attr:
            j = prev_index(pos)
            if j is not None and results[j]["author"] == default_author:
                results[j]["author"] = tidy_quote_text(m_attr.group(1))

    return [{"text": r["text"], "author": r["author"], "tags": r.get("tags", [])} for r in results]
