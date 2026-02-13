# parser_core.py

import re
import hashlib
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
_WS_NEWLINE_TRIM_RE = re.compile(r"[ \t]*\n[ \t]*")  # preserve blank lines
_WS_MULTI_RE = re.compile(r"\s+")
_PUNCT_SPACE_RE = re.compile(r"\s+([,.;:!?])")

# NOTE: preserve apostrophes to avoid collapsing "we're" vs "were"
_DEDUPE_RE = re.compile(r"[\s\"“”‘’`]+")


def normalize_ws(s: str) -> str:
    """
    Normalize whitespace while preserving paragraph breaks.
    Keeps blank lines intact (critical for paragraph-scoped attribution + paragraph quote scanning).
    """
    s = s.replace("\u00a0", " ")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _WS_SPACES_RE.sub(" ", s)
    s = _WS_NEWLINE_TRIM_RE.sub("\n", s)
    return s.strip()


def dataset_key(text: str) -> str:
    """
    Canonical key used for dataset-level dedupe (JSONL) and safe matching.
    Preserves apostrophes (don't collapse contractions).
    """
    t = normalize_ws(text).lower().strip(" \t\r\n\"“”‘’`")  # intentionally NOT stripping '
    t = _WS_MULTI_RE.sub(" ", t)
    return t


# Backwards-compatible name used by app.py
def normalize_key(text: str) -> str:
    return dataset_key(text)


def dedupe_key(s: str) -> str:
    """
    In-parse dedupe key. Slightly looser than dataset_key, but still preserves apostrophes.
    """
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


def stable_group_id(seed: str) -> str:
    return hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()[:12]


# strip quoted segments so name inference doesn't grab names from prior quotes
_QUOTED_SEG_RE = re.compile(r"[“\"«].+?[”\"»]", re.DOTALL)


def strip_quoted_segments(s: str) -> str:
    return _QUOTED_SEG_RE.sub(" ", s)


# -----------------------------
# Speech-like / scare-quote filtering
# -----------------------------

_SPEECH_HINT_RE = re.compile(
    r"\b(i|we|you|my|our|me|us|i'm|we're|you're|don't|can't|won't|it's|that's|there's)\b",
    re.IGNORECASE,
)


def speech_like_quote(q: str) -> bool:
    t = q.strip()
    if not t:
        return False
    if any(p in t for p in (".", "?", "!")):
        return True
    if _SPEECH_HINT_RE.search(t):
        return True
    return False


# -----------------------------
# Regex: noise + structural markers
# -----------------------------

GOODREADS_TAGS_RE = re.compile(r"^\s*tags\s*:\s*(.+?)\s*$", re.IGNORECASE)
LIKES_RE = re.compile(r"^\s*\d+\s+likes\s*$", re.IGNORECASE)
LIKE_WORD_RE = re.compile(r"^\s*Like\s*$", re.IGNORECASE)
NUMBER_HEADER_RE = re.compile(r"^\s*Number\s+\w+\s*:\s*(?:\(.+?\))?\s*$", re.IGNORECASE)
ATTRIBUTION_LINE_RE = re.compile(r"^\s*(?:—|―|-)\s*([^,\n]{2,120})(?:,.*)?\s*$")
SPEAKER_LABEL_RE = re.compile(r"^\s*([A-Z][A-Z0-9_ \-]{1,24})\s*:\s*(.+?)\s*$")

TITLE_SPEAKER_LABEL_RE = re.compile(r"^\s*([A-Z][A-Za-z.'\- ]{1,40})\s*:\s*(.+?)\s*$")
BAD_SPEAKER_LABELS = {
    "Transcript", "Advertisement", "Sponsored", "Related", "Read More", "More", "Note",
    "Sign Up", "Newsletter", "Latest", "Breaking", "Update",
    "Video", "Live", "Opinion", "Analysis", "Watch", "Listen", "Fact Check",
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

BULLET_RE = re.compile(r"^\s*(?:[-*•‣▪]|(\d+)[.)])\s+")
QUOTEY_LINE_RE = re.compile(r"[\"“”]")

IMAGE_CREDIT_LINE_RE = re.compile(
    r".+\s*[—–-]\s*(Getty Images|AP|Associated Press|Reuters|Shutterstock|AFP|Alamy)\s*$",
    re.IGNORECASE,
)
PHOTO_CREDIT_TAIL_RE = re.compile(
    r"(?:—|–|-)\s*(Getty Images|AP|Associated Press|Reuters|Shutterstock|AFP|Alamy)\b.*",
    re.IGNORECASE,
)
PHOTO_CREDIT_SEG_RE = re.compile(
    r"\b[A-Z][A-Za-z.'\- ]{1,60}\s*[—–-]\s*(Getty Images|AP|Associated Press|Reuters|Shutterstock|AFP|Alamy)\b",
    re.IGNORECASE,
)
CREDIT_WORD_RE = re.compile(
    r"\b(getty|reuters|associated press|ap|afp|alamy|shutterstock)\b",
    re.IGNORECASE,
)

GERUND_SINGLE_RE = re.compile(r"^[A-Z][a-z]{2,}ing$")


def scrub_attrib_context(s: str) -> str:
    s = PHOTO_CREDIT_SEG_RE.sub("", s)
    s = PHOTO_CREDIT_TAIL_RE.sub("", s)
    return s


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
    if IMAGE_CREDIT_LINE_RE.match(l):
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


# -----------------------------
# Attribution (names + verbs + orgs)
# -----------------------------

# UPDATED: allow internal-cap segments like McNeill, MacArthur, DeSantis
NAME_WORD = r"[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+(?:[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+)*(?:[-'][A-ZÀ-ÖØ-Þa-zà-öø-ÿ]+)?"
INITIALS = r"(?:[A-Z]\.){1,3}"
PARTICLE = r"(?:de|del|da|di|la|le|van|von|der|den|du|st)\.?"
SUFFIX = r"(?:Jr\.|Sr\.|II|III|IV)"

NAME_TOKEN = rf"(?:{NAME_WORD}|{INITIALS})"
NAME_PHRASE = rf"{NAME_TOKEN}(?:\s+(?:{PARTICLE}\s+)?{NAME_TOKEN}){{0,4}}(?:\s+{SUFFIX})?"

FULLNAME_RE = re.compile(
    rf"\b({NAME_TOKEN})(?:\s+(?:{PARTICLE}\s+)?({NAME_TOKEN}))?"
    rf"(?:\s+(?:{PARTICLE}\s+)?({NAME_TOKEN}))?(?:\s+(?:{PARTICLE}\s+)?({NAME_TOKEN}))?"
    rf"(?:\s+({SUFFIX}))?\b"
)
HONORIFIC_RE = re.compile(r"^(Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.)\s+", re.IGNORECASE)

ATTR_VERB_RE = (
    r"(?:according to|said|says|tell|tells|told|wrote|write|writes|stated|states|notes|noted|argued|added|"
    r"explained|joked|quipped|teased|continued|recalled|insisted|admitted|warned|"
    r"posted|tweeted|shared|claimed|alleged|alleges|accused|accuses|"
    r"said in a statement|wrote in a statement|told reporters|told (?:abc7|cnn|bbc|reuters|ap|time|the times|"
    r"the post|the guardian|fox news|nbc|cbs|msnbc|npr))"
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
    re.compile(
        rf"^\s*[,–—-]?\s*(the group|the agency|the department|the committee|the organization|the organisation)\s+(?:{ATTR_VERB_RE})\b",
        re.IGNORECASE,
    ),
    re.compile(rf"^\s*[,–—-]?\s*({ORG_PHRASE_RE.pattern})\s+(?:{ATTR_VERB_RE})\b", re.IGNORECASE),
]

BEFORE_ORG_ATTR_PATTERNS = [
    re.compile(rf"({ORG_PHRASE_RE.pattern})\s+(?:{ATTR_VERB_RE}).{{0,160}}[:;,]?\s*$", re.IGNORECASE),
    re.compile(
        rf"\b(the group|the agency|the department|the committee|the organization|the organisation)\s+(?:{ATTR_VERB_RE}).{{0,160}}[:;,]?\s*$",
        re.IGNORECASE,
    ),
]

GENERIC_ROLE_SAID_RE = re.compile(
    rf"\bthe (?:musician|singer|rapper|artist|actor|comedian|athlete|president|spokesperson|official|lawyer|judge)\s+(?:{ATTR_VERB_RE})\b",
    re.IGNORECASE,
)

# UPDATED: allow "X of South Dakota said" while still capturing X
REPORTING_SPEAKER_BEFORE_RE = re.compile(
    rf"({NAME_PHRASE})(?:\s+of\s+[A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){{0,2}})?\s*,?[^“”\"«»]{{0,240}}?\b(?:{ATTR_VERB_RE})\b",
    re.IGNORECASE,
)

SPOKESPERSON_FOR_OFFICE_RE = re.compile(
    rf"\b(?:a\s+)?spokesperson\s+for\s+({NAME_PHRASE})\s*'?\s*office\b.*?\b(?:{ATTR_VERB_RE})\b",
    re.IGNORECASE,
)
OFFICE_SAID_RE = re.compile(
    rf"\b({NAME_PHRASE})'s\s+office\s+(?:{ATTR_VERB_RE})\b",
    re.IGNORECASE,
)

AFTER_ATTR_PATTERNS = [
    re.compile(
        rf'^\s*(?:and|or)\s+(?:an?\s+|the\s+)?[“"](?P<q2>.+?)[”"]\s*[,–—-]?\s*(?P<n>{NAME_PHRASE})\s+(?:{ATTR_VERB_RE})\b',
        re.IGNORECASE,
    ),
    re.compile(rf"^\s*[,–—-]?\s*({NAME_PHRASE})\s+(?:{ATTR_VERB_RE})\b", re.IGNORECASE),
    re.compile(rf"^\s*[,–—-]?\s*(?:{ATTR_VERB_RE})\s+({NAME_PHRASE})\b", re.IGNORECASE),
    re.compile(rf"^\s*[,–—-]?\s*({ROLE_PREFIX}{NAME_PHRASE})\s+(?:{ATTR_VERB_RE})\b", re.IGNORECASE),
    re.compile(rf"^\s*[,–—-]?\s*({NAME_PHRASE})\s*,[^.]*?\b(?:{ATTR_VERB_RE})\b", re.IGNORECASE),
    re.compile(rf"^\s*[,–—-]?\s*[—–-]\s*({ORG_PHRASE_RE.pattern})\b", re.IGNORECASE),
    re.compile(r"^\s*[,–—-]?\s*(their statement|they said|the statement)\s+(continued|said)\b", re.IGNORECASE),
]

BEFORE_ATTR_PATTERNS = [
    re.compile(rf"({NAME_PHRASE})\s+(?:{ATTR_VERB_RE})[^“”\"]{{0,200}}[:;,]?\s*$", re.IGNORECASE),
    re.compile(rf"({NAME_PHRASE})\s*,[^,\n]{{0,200}}?,\s*(?:{ATTR_VERB_RE})[^“”\"]{{0,200}}[:;,]?\s*$", re.IGNORECASE),
]

# -----------------------------
# Document/report attribution
# -----------------------------

DOC_SUBJECT = (
    r"(?:the\s+(?:report|study|paper|review|analysis|audit)|this\s+(?:report|study|paper|review|analysis)|"
    r"the\s+authors|the\s+author|the\s+researchers|the\s+team|the\s+group)"
)
DOC_VERB = r"(?:says|said|notes|noted|finds|found|warns|warned|concludes|concluded|adds|added|argues|argued|states|stated|writes|wrote|observes|observed|reports|reported|explains|explained|concedes|conceded)"

AFTER_DOC_ATTR_PATTERNS = [
    re.compile(rf"^\s*[,–—-]?\s*({DOC_SUBJECT})\s+(?:{DOC_VERB})\b", re.IGNORECASE),
    re.compile(rf"^\s*[,–—-]?\s*(it)\s+(?:{DOC_VERB})\b", re.IGNORECASE),
]
BEFORE_DOC_ATTR_PATTERNS = [
    re.compile(rf"\b({DOC_SUBJECT})\s+(?:{DOC_VERB}).{{0,200}}[:;,]?\s*$", re.IGNORECASE),
    re.compile(rf"\b(it)\s+(?:{DOC_VERB}).{{0,200}}[:;,]?\s*$", re.IGNORECASE),
]

DOC_TITLE_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&.\-']+(?:\s+[A-Z][A-Za-z0-9&.\-']+){0,10}\s+(?:Report|Study|Paper|Review|Analysis|Audit))\b"
)

def infer_document_title(context_before: str, context_after: str) -> Optional[str]:
    before = fast_context_norm(context_before)[-CTX_WINDOW:]
    after = fast_context_norm(context_after)[:CTX_WINDOW]
    ctx = f"{before} {after}"
    ms = list(DOC_TITLE_RE.finditer(ctx))
    if ms:
        return ms[-1].group(1).strip()
    m2 = re.search(r"\b(International AI Safety Report)\b", ctx)
    if m2:
        return m2.group(1)
    return None


def normalize_doc_subject(subject: str, context_before: str, context_after: str) -> str:
    title = infer_document_title(context_before, context_after)
    if title:
        return title
    s = normalize_ws(subject).strip().lower()
    if s == "it":
        return "Document"
    return s[:1].upper() + s[1:]


# -----------------------------
# Person/author filtering
# -----------------------------

# US states + DC (to stop "South Dakota"/"Massachusetts" etc from being speakers)
_US_STATES = {
    "alabama","alaska","arizona","arkansas","california","colorado","connecticut","delaware","florida","georgia",
    "hawaii","idaho","illinois","indiana","iowa","kansas","kentucky","louisiana","maine","maryland","massachusetts",
    "michigan","minnesota","mississippi","missouri","montana","nebraska","nevada","new hampshire","new jersey","new mexico",
    "new york","north carolina","north dakota","ohio","oklahoma","oregon","pennsylvania","rhode island","south carolina",
    "south dakota","tennessee","texas","utah","vermont","virginia","washington","west virginia","wisconsin","wyoming",
    "district of columbia","washington, d.c.","dc","d.c.",
}

# common group nouns that show up capitalized near attributions
_GROUP_NOUNS = {
    "democrats","republicans","lawmakers","senators","governors","leaders","officials","sources","aides",
    "the democrats","the republicans","the white house",
}

NON_NAME_SINGLETONS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
    "the", "a", "an",
    "times", "post", "reuters", "ap", "bbc", "cnn", "court", "police", "judge", "officials", "source", "sources",
    "images", "getty",
    "spanish", "english", "french", "german", "italian", "portuguese", "arabic", "chinese", "russian", "japanese", "korean",
    "immigration", "customs", "enforcement", "ice",
    "grammy", "grammys", "awards", "super", "bowl", "halftime", "show", "apple", "music",
    "calling", "saying", "asking", "warning", "noting", "adding", "explaining",
    "on",
    # NEW: party/group nouns
    "democrats","republicans","lawmakers","senators","governors","leaders","officials",
}

NON_PERSON_LAST_TOKENS = {
    "enforcement", "department", "agency", "administration", "committee", "council",
    "office", "university", "hospital", "ministry", "commission", "foundation",
    "organization", "organisation", "service", "services",
    "awards", "award", "show", "halftime", "performance", "language", "focus",
    "house",
}

TITLE_PSEUDO_NAMES = {
    "the president", "the vice president", "the white house", "the administration", "the government",
    "the spokesperson", "the press secretary", "the musician", "the singer", "the rapper", "the artist",
    "the actor", "the comedian", "the athlete", "the judge", "the court",
    "president", "vice president", "senator", "governor", "mayor", "speaker", "chair",
    "commissioner", "judge", "press secretary", "spokesperson",
    "white house",
}

EVENT_PHRASES = {
    "grammy awards", "the grammy awards", "super bowl", "super bowl halftime show", "halftime show",
    "all american halftime show",
}

_WEEKDAY_RE = re.compile(r"^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$", re.IGNORECASE)
_MONTH_RE = re.compile(
    r"^(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?$",
    re.IGNORECASE,
)
_ON_DATE_LEADIN_RE = re.compile(
    r"^on\s+([A-Za-z]{3,9}|[A-Za-z]{3,9}\.\s+\d{1,2}|\d{1,2})\b",
    re.IGNORECASE,
)

# strip discourse lead-ins that often precede a real name in news prose
_DISCOURSE_PREFIX_RE = re.compile(
    r"^\s*(?:But|However|Meanwhile|Still|So|And|Then|Later)\s+",
    re.IGNORECASE,
)


def _looks_like_date_leadin(name: str) -> bool:
    n = normalize_ws(name).strip()
    low = n.lower()
    if low.startswith("on "):
        rest = low[3:].strip()
        if _WEEKDAY_RE.match(rest):
            return True
        first = rest.split()[0]
        if _WEEKDAY_RE.match(first) or _MONTH_RE.match(first):
            return True
        if _ON_DATE_LEADIN_RE.match(low):
            return True
    return False


def _is_geo_phrase(name: str) -> bool:
    n = normalize_ws(name).strip().lower()
    return n in _US_STATES


def _is_group_noun(name: str) -> bool:
    n = normalize_ws(name).strip().lower()
    return n in _GROUP_NOUNS


def _is_non_person_name(name: str) -> bool:
    parts = [p.strip(".") for p in name.split() if p.strip(".")]
    if not parts:
        return True
    last = parts[-1].lower()
    return last in NON_PERSON_LAST_TOKENS


def _is_title_pseudo_name(name: str) -> bool:
    n = normalize_ws(name).lower()
    if n in TITLE_PSEUDO_NAMES:
        return True
    if n.startswith("the "):
        return n[4:] in TITLE_PSEUDO_NAMES
    return False


def clean_author_candidate(a: str) -> Optional[str]:
    if not a:
        return None

    n0 = tidy_quote_text(a)
    n0 = PHOTO_CREDIT_TAIL_RE.sub("", n0).strip()
    if not n0:
        return None

    # drop discourse lead-ins ("But Trump" -> "Trump")
    n = _DISCOURSE_PREFIX_RE.sub("", n0).strip()

    if _looks_like_date_leadin(n):
        return None

    # NEW: reject geo phrases + group nouns early
    if _is_geo_phrase(n):
        return None
    if _is_group_noun(n):
        return None

    low = n.lower().strip(".,;:!?")

    if " " not in n and low in {"the", "a", "an"}:
        return None
    if CREDIT_WORD_RE.search(n):
        return None
    if _is_title_pseudo_name(n):
        return None
    if _is_non_person_name(n):
        return None
    if " " not in n and low in NON_NAME_SINGLETONS:
        return None
    if " " not in n and GERUND_SINGLE_RE.match(n) and low not in {"king"}:
        return None
    for ev in EVENT_PHRASES:
        if ev in low:
            return None
    return n


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


def extract_best_person_name(s: str) -> Optional[str]:
    """
    NEW: choose the BEST candidate, not simply the last one.
    Prefer multi-token person-like names; penalize geo phrases and "of <Place>" tails.
    """
    s = normalize_ws(s)
    honorific = ""
    m_h = HONORIFIC_RE.match(s)
    if m_h:
        honorific = m_h.group(1).strip() + " "
        s = HONORIFIC_RE.sub("", s, count=1)

    matches = list(FULLNAME_RE.finditer(s))
    best_name: Optional[str] = None
    best_score = -10_000

    for mm in matches:
        parts = [p for p in mm.groups() if p]
        parts = [p for p in parts if p and p.lower().strip(".") not in {"de", "del", "da", "di", "la", "le", "van", "von", "der", "den", "du", "st"}]
        if not parts:
            continue
        cand = " ".join(parts).strip()
        if not cand:
            continue

        # heuristics
        tok_n = len(cand.split())
        score = 10 * tok_n

        # penalize if preceded by "of " (common location tail)
        pre = s[max(0, mm.start() - 4): mm.start()].lower()
        if pre.endswith("of "):
            score -= 7

        # penalize geo or group nouns hard
        if _is_geo_phrase(cand):
            score -= 50
        if _is_group_noun(cand):
            score -= 50

        # slight bonus if multi-token
        if tok_n >= 2:
            score += 5

        if score > best_score:
            best_score = score
            best_name = cand

    if best_name:
        name = (honorific + best_name).strip()
        return clean_author_candidate(name)

    # fallback: single token (still filtered)
    m = re.search(rf"\b({NAME_TOKEN})\b", s)
    if not m:
        return None
    name = (honorific + m.group(1)).strip()
    if name.lower().strip(".") in {"the", "a", "an"}:
        return None
    return clean_author_candidate(name)


def infer_group_author(context_before: str) -> Optional[str]:
    ctx = fast_context_norm(context_before)[-GROUP_CTX_WINDOW:]
    m = re.search(rf"\b({NAME_TOKEN})\s+and\s+({NAME_TOKEN})\s+({NAME_TOKEN})\b", ctx)
    if m:
        return clean_author_candidate(f"{m.group(1)} and {m.group(2)} {m.group(3)}")
    m2 = re.search(rf"\b({NAME_TOKEN})\s+and\s+({NAME_TOKEN})\b", ctx)
    if m2:
        return clean_author_candidate(f"{m2.group(1)} and {m2.group(2)}")
    return None


def _followed_by_and_cap(text: str, token: str) -> bool:
    return bool(re.search(rf"\b{re.escape(token)}\s+and\s+[A-Z]", text))


_OBJ_PREP_RE = re.compile(r"(?:\b(at|to|for|of|from|with|against)\s+)$", re.IGNORECASE)


def _is_object_of_prep(before_norm: str, start_idx: int) -> bool:
    window = before_norm[max(0, start_idx - 24): start_idx]
    return bool(_OBJ_PREP_RE.search(window))


def infer_nearest_name_in_before(before: str) -> Optional[str]:
    b = normalize_ws(strip_quoted_segments(before))
    b_low = b.lower()

    matches = list(FULLNAME_RE.finditer(b))
    for mm in reversed(matches):
        if _is_object_of_prep(b_low, mm.start()):
            continue
        parts = [p for p in mm.groups() if p]
        parts = [p for p in parts if p and p.lower().strip(".") not in {"de", "del", "da", "di", "la", "le", "van", "von", "der", "den", "du", "st"}]
        cand = " ".join(parts).strip()
        cand2 = clean_author_candidate(cand)
        if cand2:
            return cand2

    tokens = list(re.finditer(rf"\b({NAME_TOKEN})\b", b))
    for mm in reversed(tokens):
        if _is_object_of_prep(b_low, mm.start()):
            continue
        cand = mm.group(1)
        if cand.lower().strip(".") in NON_NAME_SINGLETONS:
            continue
        if _followed_by_and_cap(b, cand):
            continue
        cand2 = clean_author_candidate(cand)
        if cand2:
            return cand2

    return None


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
    return bool(re.search(r"\b(podcast|episode|show|series)\b", before, re.IGNORECASE))


def _upgrade_from_carry_if_prefix(author: str, carry: Optional[str]) -> str:
    if not author or not carry:
        return author
    a = normalize_ws(author).strip()
    c = normalize_ws(carry).strip()
    if not c or " " not in c:
        return author
    if " " in a:
        return author
    if c.lower().startswith(a.lower() + " "):
        return c
    return author


def resolve_author_for_quote(
    context_before: str,
    context_after: str,
    default_author: str,
    lastname_map: Optional[Dict[str, str]],
    last_known_author_in_paragraph: Optional[str],
) -> Tuple[str, str]:
    before_full = scrub_attrib_context(fast_context_norm(context_before)[-CTX_WINDOW:])
    after = scrub_attrib_context(fast_context_norm(context_after)[:CTX_WINDOW])

    cb = context_before.replace("\r\n", "\n").replace("\r", "\n")
    cut = cb.rfind("\n\n")
    local_before_raw = cb[cut + 2:] if cut != -1 else cb
    before_local = scrub_attrib_context(fast_context_norm(local_before_raw)[-400:])

    before = before_full

    m_off = OFFICE_SAID_RE.search(before_local)
    if m_off:
        who = extract_best_person_name(m_off.group(1)) or normalize_ws(m_off.group(1))
        who = clean_author_candidate(who) or who
        if who:
            return f"{who}'s office", "before_office"

    m_sp = SPOKESPERSON_FOR_OFFICE_RE.search(before_local)
    if m_sp:
        who = extract_best_person_name(m_sp.group(1)) or normalize_ws(m_sp.group(1))
        who = clean_author_candidate(who) or who
        if who:
            return f"{who}'s office", "before_office"

    before_nq = strip_quoted_segments(before)
    reps = list(REPORTING_SPEAKER_BEFORE_RE.finditer(before_nq))

    if reps:
        tail = before_local[-240:].lower()
        if "spokesperson for" in tail or "'s office" in tail:
            reps = []

    if reps:
        cand = extract_best_person_name(reps[-1].group(1))
        cand = clean_author_candidate(cand or "")
        if cand:
            cand = _upgrade_from_carry_if_prefix(cand, last_known_author_in_paragraph)
        if cand and " " not in cand and _followed_by_and_cap(before, cand):
            cand = None
        if cand:
            if lastname_map and len(cand.split()) == 1 and cand in lastname_map:
                cand2 = clean_author_candidate(lastname_map[cand]) or cand
                return cand2, "before_reporting_speaker"
            return cand, "before_reporting_speaker"

    for pat in AFTER_DOC_ATTR_PATTERNS:
        m = pat.search(after)
        if m:
            subj = m.group(1)
            return normalize_doc_subject(subj, before, after), "after_document"
    for pat in BEFORE_DOC_ATTR_PATTERNS:
        m = pat.search(before)
        if m:
            subj = m.group(1)
            return normalize_doc_subject(subj, before, after), "before_document"

    if re.search(r"\b(their statement|they said|the statement)\b", after, re.IGNORECASE):
        g = infer_group_author(before)
        if g:
            return g, "group_infer"

    if PRONOUN_AFTER_RE.search(after):
        inferred = infer_nearest_name_in_before(before)
        if inferred:
            inferred = _upgrade_from_carry_if_prefix(inferred, last_known_author_in_paragraph)
            if lastname_map and len(inferred.split()) == 1 and inferred in lastname_map:
                inferred2 = clean_author_candidate(lastname_map[inferred]) or inferred
                return inferred2, "after_pronoun"
            return inferred, "after_pronoun"

    if GENERIC_ROLE_SAID_RE.search(after) or GENERIC_ROLE_SAID_RE.search(before):
        inferred = infer_nearest_name_in_before(before)
        if inferred:
            inferred = _upgrade_from_carry_if_prefix(inferred, last_known_author_in_paragraph)
            if lastname_map and len(inferred.split()) == 1 and inferred in lastname_map:
                inferred2 = clean_author_candidate(lastname_map[inferred]) or inferred
                return inferred2, "generic_role_infer"
            return inferred, "generic_role_infer"

    for pat in AFTER_ORG_ATTR_PATTERNS:
        m = pat.search(after)
        if not m:
            continue
        src = m.group(1).strip()
        if src.lower() in GENERIC_ORG_WORDS:
            org = infer_nearest_org_in_before(before)
            if org:
                return org, "after_org_infer"
            carry = clean_author_candidate(last_known_author_in_paragraph or "")
            return (carry or default_author), "carry_or_default"
        return src, "after_org"

    m_shared = AFTER_ATTR_PATTERNS[0].search(after)
    if m_shared:
        name = extract_best_person_name(m_shared.group("n"))
        if name:
            name = _upgrade_from_carry_if_prefix(name, last_known_author_in_paragraph)
            if lastname_map and len(name.split()) == 1 and name in lastname_map:
                name2 = clean_author_candidate(lastname_map[name]) or name
                return name2, "after_shared_name"
            return name, "after_shared_name"

    for pat in AFTER_ATTR_PATTERNS[1:]:
        m = pat.search(after)
        if not m:
            continue
        name = extract_best_person_name(m.group(1))
        if name:
            name = _upgrade_from_carry_if_prefix(name, last_known_author_in_paragraph)
            if lastname_map and len(name.split()) == 1 and name in lastname_map:
                name2 = clean_author_candidate(lastname_map[name]) or name
                return name2, "after_name"
            return name, "after_name"

    for pat in BEFORE_ORG_ATTR_PATTERNS:
        m = pat.search(before)
        if not m:
            continue
        src = m.group(1).strip()
        if src.lower() in GENERIC_ORG_WORDS:
            org = infer_nearest_org_in_before(before)
            if org:
                return org, "before_org_infer"
            carry = clean_author_candidate(last_known_author_in_paragraph or "")
            return (carry or default_author), "carry_or_default"
        return src, "before_org"

    for pat in BEFORE_ATTR_PATTERNS:
        m = pat.search(before)
        if not m:
            continue
        name = extract_best_person_name(m.group(1))
        if name:
            name = _upgrade_from_carry_if_prefix(name, last_known_author_in_paragraph)
            if lastname_map and len(name.split()) == 1 and name in lastname_map:
                name2 = clean_author_candidate(lastname_map[name]) or name
                return name2, "before_name"
            return name, "before_name"

    carry = clean_author_candidate(last_known_author_in_paragraph or "")
    if carry:
        return carry, "carry_or_default"
    return default_author, "carry_or_default"


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
    include_debug: bool,
    enable_speech_filter: bool,
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
        q_full = tidy_quote_text(q_raw)
        candidates = chunk_quote_to_maxlen(q_raw, max_len=max_len) if len(q_full) > max_len else [q_raw]
        group_id = stable_group_id(f"inline|{q_full}|{line_starts[i]}") if len(candidates) > 1 else ""

        author_raw = m.group("a") or m.group("a2") or default_author
        author = clean_author_candidate(author_raw) or default_author

        for ci, c in enumerate(candidates):
            qt = clamp_minimal(c, min_len, max_len, max_newlines, max_sentences)
            if not qt:
                continue
            if enable_speech_filter and not speech_like_quote(qt):
                continue
            rec: Dict[str, object] = {"text": qt, "author": author, "tags": [], "_pos": line_starts[i]}
            if include_debug:
                rec["_mode"] = "inline"
                rec["_kind"] = "speech"
                if group_id:
                    rec["_group_id"] = group_id
                    rec["_chunk_i"] = ci + 1
                    rec["_chunk_n"] = len(candidates)
            out.append(rec)
    return out


# -----------------------------
# Quoted-span scanning
# -----------------------------

_PARA_SPLIT_RE = re.compile(r"\n\s*\n")


def scan_quote_spans(text: str) -> List[Tuple[int, int]]:
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
        if i <= 0 or i + 1 >= len(text):
            return False
        return is_word_char(text[i - 1]) and is_word_char(text[i + 1])

    def looks_like_dangling_closer(i: int) -> bool:
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
# Main extraction
# -----------------------------

def _author_upgrade(old_author: str, new_author: str, default_author: str) -> bool:
    if not new_author:
        return False
    if (
        clean_author_candidate(new_author) is None
        and new_author not in {default_author, "Document"}
        and not new_author.lower().endswith(("report", "study", "paper", "review", "analysis", "audit", "office"))
    ):
        return False
    if old_author == default_author and new_author != default_author:
        return True
    if old_author != default_author and new_author != default_author:
        return len(new_author) > len(old_author)
    return False


def extract_quotes(
    text: str,
    default_author: str = "Unknown",
    *,
    min_len: int = DEFAULT_MIN_LEN,
    max_len: int = DEFAULT_MAX_LEN,
    max_newlines: int = DEFAULT_MAX_NEWLINES,
    max_sentences: int = DEFAULT_MAX_SENTENCES,
    enable_inline_attribution: bool = True,
    enable_quoted_spans: bool = True,
    enable_dialogue_lines: bool = False,
    enable_quote_collections: bool = False,
    enable_tables: bool = False,
    enable_paragraph_attribution: bool = True,
    enable_speech_filter: bool = False,
    include_debug: bool = False,
) -> List[Dict[str, object]]:
    raw = normalize_ws(text)
    line_starts = build_line_starts(raw)

    results: List[Dict[str, object]] = []
    seen_map: Dict[str, int] = {}

    para_ranges: List[Tuple[int, int]] = []
    idx = 0
    for m in re.finditer(r"\n\s*\n", raw):
        para_ranges.append((idx, m.start()))
        idx = m.end()
    para_ranges.append((idx, len(raw)))
    para_starts = [a for a, _ in para_ranges]

    def paragraph_index_for_pos(pos: int) -> int:
        return max(0, bisect_right(para_starts, pos) - 1)

    # Store (author, last_attrib_pos) -- store quote END
    last_author_by_para: Dict[int, Tuple[str, int]] = {}

    lastname_map: Optional[Dict[str, str]] = None
    lastname_map_ready = False

    def get_lastname_map() -> Dict[str, str]:
        nonlocal lastname_map, lastname_map_ready
        if not lastname_map_ready:
            lastname_map = build_lastname_map(raw)
            lastname_map_ready = True
        return lastname_map or {}

    def add_or_upgrade(record: Dict[str, object]) -> None:
        k = dedupe_key(str(record.get("text", "")))
        if not k:
            return
        existing_i = seen_map.get(k)
        if existing_i is None:
            seen_map[k] = len(results)
            results.append(record)
            return
        old = results[existing_i]
        old_author = str(old.get("author", default_author))
        new_author = str(record.get("author", default_author))
        if _author_upgrade(old_author, new_author, default_author):
            old["author"] = new_author
        if record.get("_author_src") and not old.get("_author_src"):
            old["_author_src"] = record["_author_src"]
        if record.get("_mode") and not old.get("_mode"):
            old["_mode"] = record["_mode"]
        if record.get("_kind") and not old.get("_kind"):
            old["_kind"] = record["_kind"]

    if enable_inline_attribution:
        for r in extract_inline_quote_attribution_lines(
            raw,
            default_author=default_author,
            min_len=min_len,
            max_len=max_len,
            max_newlines=max_newlines,
            max_sentences=max_sentences,
            line_starts=line_starts,
            include_debug=include_debug,
            enable_speech_filter=enable_speech_filter,
        ):
            add_or_upgrade(r)

    spans = scan_quote_spans(raw) if enable_quoted_spans else []
    if spans:
        lm = get_lastname_map() if len(spans) >= 2 else None
        for (s, e) in spans:
            inside = raw[s + 1: e]
            if not inside.strip():
                continue

            before = raw[max(0, s - CTX_WINDOW): s]
            after = raw[e + 1: min(len(raw), e + 1 + CTX_WINDOW)]

            candidates = chunk_quote_to_maxlen(inside, max_len=max_len) if len(tidy_quote_text(inside)) > max_len else [inside]

            para_i = paragraph_index_for_pos(s)
            carry_author: Optional[str] = None
            if enable_paragraph_attribution:
                carry = last_author_by_para.get(para_i)
                if carry:
                    a, a_pos = carry
                    if (s - a_pos) <= MAX_CARRY_DISTANCE:
                        carry_author = a

            author, author_src = resolve_author_for_quote(
                before,
                after,
                default_author=default_author,
                lastname_map=lm,
                last_known_author_in_paragraph=carry_author,
            )

            author_clean = clean_author_candidate(author) or author or default_author
            if carry_author:
                author_clean = _upgrade_from_carry_if_prefix(author_clean, carry_author)

            if enable_paragraph_attribution and author_clean != default_author:
                last_author_by_para[para_i] = (author_clean, e)

            for cand in candidates:
                cleaned = clamp_minimal(cand, min_len, max_len, max_newlines, max_sentences)
                if not cleaned:
                    continue
                if enable_speech_filter and not speech_like_quote(cleaned):
                    continue
                rec: Dict[str, object] = {"text": cleaned, "author": author_clean, "tags": [], "_pos": s}
                if include_debug:
                    rec["_mode"] = "spans"
                    rec["_author_src"] = author_src
                    rec["_kind"] = "speech"
                add_or_upgrade(rec)

    if include_debug:
        return [
            {
                "text": r.get("text", ""),
                "author": r.get("author", default_author),
                "tags": r.get("tags", []),
                "_mode": r.get("_mode", ""),
                "_kind": r.get("_kind", ""),
                "_author_src": r.get("_author_src", ""),
            }
            for r in results
        ]

    return [{"text": r["text"], "author": r["author"], "tags": r.get("tags", [])} for r in results]


# -----------------------------
# Tag parsing (used by app.py)
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
