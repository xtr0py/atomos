import os
import json
import hashlib
from typing import List, Dict, Set
from collections import Counter

import pandas as pd
import streamlit as st

from parser_core import (
    extract_quotes,
    normalize_key,     # now unified with dataset_key in parser_core
    parse_tag_line,
    DEFAULT_MIN_LEN,
    DEFAULT_MAX_LEN,
    DEFAULT_MAX_NEWLINES,
    DEFAULT_MAX_SENTENCES,
)

st.set_page_config(page_title="Quote Parser (JSONL)", layout="wide")
st.title("Quote Parser")
st.caption("Upload a .txt or paste text. Curate minimal {text, author, tags[]} entries into JSONL (append-only).")

DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", "/data")
DEFAULT_JSONL_PATH = os.environ.get("JSONL_PATH", os.path.join(DEFAULT_DATA_DIR, "quotes.jsonl"))

# -----------------------------
# Helpers
# -----------------------------

def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def tags_to_str(tags: List[str]) -> str:
    return ", ".join(tags or [])

@st.cache_data(show_spinner=False)
def load_existing_keys(path: str) -> Set[str]:
    keys: Set[str] = set()
    if not os.path.exists(path):
        return keys
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "text" in obj:
                        keys.add(normalize_key(str(obj["text"])))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return set()
    return keys

def append_jsonl(path: str, rows: List[Dict[str, object]]) -> int:
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)

# -----------------------------
# Presets + UX copy
# -----------------------------

PRESET_ORDER = [
    "📰 News article (recommended)",
    "🎙 Transcript / Interview",
    "📜 Curated quote list (quote pages / Goodreads)",
    "📊 Table import (quote + author columns)",
    "⚙ Custom",
]

PRESET_LONG_HELP: Dict[str, str] = {
    "📰 News article (recommended)":
        "Use for standard articles or blog posts that mix narrative text with quoted speech.\n"
        "Automatically finds quoted text, infers speakers from nearby “X said” patterns, and filters out many editorial “scare quotes.”",

    "🎙 Transcript / Interview":
        "Use for interviews, hearings, or transcripts where speakers are labeled (e.g., HOST:, NEWSOM:, Q: / A:).\n"
        "Prioritizes speaker labels and avoids extracting incidental quoted phrases.",

    "📜 Curated quote list (quote pages / Goodreads)":
        "Use for pages that are primarily lists of quotes, often with bullets, separators, author lines, or tags.\n"
        "Optimized for quote databases and Goodreads-style pages.",

    "📊 Table import (quote + author columns)":
        "Use when text is already structured into columns, such as tab-separated or space-aligned rows with quote and author fields.\n"
        "Ignores prose parsing for maximum precision.",

    "⚙ Custom":
        "Use when none of the presets fit your input, or when you want full manual control over parsing behavior.",
}

PRESETS: Dict[str, Dict[str, bool]] = {
    "📰 News article (recommended)": {
        "enable_inline_attribution": True,
        "enable_quoted_spans": True,
        "enable_dialogue_lines": False,
        "enable_quote_collections": False,
        "enable_tables": False,
        "enable_paragraph_attribution": True,
        "enable_speech_filter": True,
    },
    "🎙 Transcript / Interview": {
        "enable_inline_attribution": False,
        "enable_quoted_spans": False,
        "enable_dialogue_lines": True,
        "enable_quote_collections": False,
        "enable_tables": False,
        "enable_paragraph_attribution": False,
        "enable_speech_filter": True,
    },
    "📜 Curated quote list (quote pages / Goodreads)": {
        "enable_inline_attribution": True,
        "enable_quoted_spans": False,
        "enable_dialogue_lines": False,
        "enable_quote_collections": True,
        "enable_tables": False,
        "enable_paragraph_attribution": False,
        "enable_speech_filter": False,
    },
    "📊 Table import (quote + author columns)": {
        "enable_inline_attribution": False,
        "enable_quoted_spans": False,
        "enable_dialogue_lines": False,
        "enable_quote_collections": False,
        "enable_tables": True,
        "enable_paragraph_attribution": False,
        "enable_speech_filter": False,
    },
    "⚙ Custom": {},
}

def apply_preset(preset_name: str) -> None:
    if preset_name not in PRESETS or preset_name == "⚙ Custom":
        return
    cfg = PRESETS[preset_name]
    for k, v in cfg.items():
        st.session_state[k] = v

def preset_tooltip_text() -> str:
    lines: List[str] = []
    lines.append("Presets configure the parser for common text formats.")
    lines.append("")
    for name in PRESET_ORDER:
        lines.append(f"{name}")
        lines.append(PRESET_LONG_HELP.get(name, "").strip())
        lines.append("")
    lines.append("Tip: Choose Custom to manually control all toggles.")
    return "\n".join(lines).strip()

# -----------------------------
# Cached parse
# -----------------------------

@st.cache_data(show_spinner=False)
def cached_parse(
    source_text: str,
    default_author: str,
    min_len: int,
    max_len: int,
    max_newlines: int,
    max_sentences: int,
    enable_inline_attribution: bool,
    enable_quoted_spans: bool,
    enable_dialogue_lines: bool,
    enable_quote_collections: bool,
    enable_tables: bool,
    enable_paragraph_attribution: bool,
    enable_speech_filter: bool,
    include_debug: bool,
) -> List[Dict[str, object]]:
    parsed = extract_quotes(
        source_text,
        default_author=default_author,
        min_len=min_len,
        max_len=max_len,
        max_newlines=max_newlines,
        max_sentences=max_sentences,
        enable_inline_attribution=enable_inline_attribution,
        enable_quoted_spans=enable_quoted_spans,
        enable_dialogue_lines=enable_dialogue_lines,
        enable_quote_collections=enable_quote_collections,
        enable_tables=enable_tables,
        enable_paragraph_attribution=enable_paragraph_attribution,
        enable_speech_filter=enable_speech_filter,
        include_debug=include_debug,
    )

    out: List[Dict[str, object]] = []
    for r in parsed:
        tags = r.get("tags", []) or []
        rec: Dict[str, object] = {
            "approve": True,
            "text": r.get("text", ""),
            "author": r.get("author", ""),
            "tags": tags,
            "tags_str": tags_to_str(tags),
        }
        if include_debug:
            rec["_mode"] = r.get("_mode", "")
            rec["_kind"] = r.get("_kind", "")
            rec["_author_src"] = r.get("_author_src", "")
            rec["_group_id"] = r.get("_group_id", "")
            rec["_chunk_i"] = r.get("_chunk_i", "")
            rec["_chunk_n"] = r.get("_chunk_n", "")
        out.append(rec)
    return out

def as_editor_df(rows: List[Dict[str, object]], include_debug: bool) -> pd.DataFrame:
    cols = ["approve", "text", "author", "tags_str"]
    if include_debug:
        cols = ["approve", "_mode", "_kind", "_author_src", "_group_id", "_chunk_i", "_chunk_n", "text", "author", "tags_str"]

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)
    for col in cols:
        if col not in df.columns:
            df[col] = True if col == "approve" else ""
    return df[cols]

# -----------------------------
# Sidebar settings
# -----------------------------

with st.sidebar:
    st.header("Settings")

    jsonl_path = st.text_input("JSONL output path", value=DEFAULT_JSONL_PATH)
    default_author = st.text_input("Default author (when none found)", value="Unknown")

    st.subheader("Preset")
    preset = st.selectbox(
        "Choose a parsing profile",
        PRESET_ORDER,
        index=0,
        help=preset_tooltip_text(),
    )

    if "preset_last" not in st.session_state:
        st.session_state["preset_last"] = preset
        apply_preset(preset)
    elif st.session_state["preset_last"] != preset:
        st.session_state["preset_last"] = preset
        apply_preset(preset)

    st.subheader("Minimal quote filters")
    min_len = st.number_input("Min length", min_value=1, max_value=500, value=int(DEFAULT_MIN_LEN))

    max_len = st.number_input(
        "Max length",
        min_value=30,
        max_value=500,
        value=int(DEFAULT_MAX_LEN),
        help="Upper bound for extracted quote text. Long quotes are chunked when possible.",
    )

    max_sentences = st.number_input("Max sentences", min_value=1, max_value=20, value=int(DEFAULT_MAX_SENTENCES))
    max_newlines = st.number_input("Max newlines", min_value=0, max_value=10, value=int(DEFAULT_MAX_NEWLINES))

    st.subheader("Basic modes")

    enable_inline_attribution = st.toggle(
        "Inline attribution (“…” — Author)",
        value=st.session_state.get("enable_inline_attribution", True),
    )

    enable_quoted_spans = st.toggle(
        "Quoted spans (“…”)",
        value=st.session_state.get("enable_quoted_spans", True),
    )

    st.subheader("Quality")

    enable_speech_filter = st.toggle(
        "Prefer speech-like quotes (reduce scare quotes)",
        value=st.session_state.get("enable_speech_filter", True),
    )

    with st.expander("Advanced modes", expanded=(preset == "⚙ Custom")):
        enable_dialogue_lines = st.toggle(
            "Dialogue lines (LABEL: text)",
            value=st.session_state.get("enable_dialogue_lines", False),
        )

        enable_quote_collections = st.toggle(
            "Curated quote lists (bullets / Goodreads / quote pages)",
            value=st.session_state.get("enable_quote_collections", False),
        )

        enable_tables = st.toggle(
            "Table import (quote ⟂ author ⟂ ...)",
            value=st.session_state.get("enable_tables", False),
        )

        carry_disabled = not bool(enable_quoted_spans)
        enable_paragraph_attribution = st.toggle(
            "Carry author within paragraph",
            value=st.session_state.get("enable_paragraph_attribution", True),
            disabled=carry_disabled,
        )

        include_debug = st.toggle(
            "Show debug columns (mode / kind / author source / chunking)",
            value=False,
        )

        if st.button("Clear parse cache"):
            cached_parse.clear()
            st.rerun()

    st.subheader("Tag helpers")
    global_tags = st.text_input("Global tags (comma-separated, applied on save)", value="")

# Keep session_state in sync
st.session_state["enable_inline_attribution"] = bool(enable_inline_attribution)
st.session_state["enable_quoted_spans"] = bool(enable_quoted_spans)
st.session_state["enable_speech_filter"] = bool(enable_speech_filter)
st.session_state["enable_dialogue_lines"] = bool(enable_dialogue_lines)
st.session_state["enable_quote_collections"] = bool(enable_quote_collections)
st.session_state["enable_tables"] = bool(enable_tables)
st.session_state["enable_paragraph_attribution"] = bool(enable_paragraph_attribution)

# -----------------------------
# Dataset info
# -----------------------------

existing_keys = load_existing_keys(jsonl_path)
st.info(f"Current dataset: **{len(existing_keys)}** unique quote(s) detected in `{jsonl_path}` (by normalized text).")

# -----------------------------
# Input section
# -----------------------------

col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Drop a .txt file here", type=["txt"])
with col2:
    pasted = st.text_area("…or paste an excerpt here", height=240, placeholder="Paste text with quotes here…")

source_text = ""
if uploaded is not None:
    source_text = uploaded.read().decode("utf-8", errors="replace")
elif pasted.strip():
    source_text = pasted

if "rows" not in st.session_state:
    st.session_state["rows"] = []

parse_clicked = st.button("Parse quotes", type="primary", disabled=not bool(source_text.strip()))
clear_clicked = st.button("Clear results", disabled=not bool(st.session_state["rows"]))

if parse_clicked:
    st.session_state["rows"] = cached_parse(
        source_text=source_text,
        default_author=default_author,
        min_len=int(min_len),
        max_len=int(max_len),
        max_newlines=int(max_newlines),
        max_sentences=int(max_sentences),
        enable_inline_attribution=bool(enable_inline_attribution),
        enable_quoted_spans=bool(enable_quoted_spans),
        enable_dialogue_lines=bool(enable_dialogue_lines),
        enable_quote_collections=bool(enable_quote_collections),
        enable_tables=bool(enable_tables),
        enable_paragraph_attribution=bool(enable_paragraph_attribution),
        enable_speech_filter=bool(enable_speech_filter),
        include_debug=bool(include_debug),
    )

if clear_clicked:
    st.session_state["rows"] = []
    st.rerun()

rows = st.session_state["rows"]

# -----------------------------
# Review + save
# -----------------------------

if not rows:
    st.write("Upload a file or paste text, then click **Parse quotes**.")
    st.stop()

st.subheader(f"Review ({len(rows)} found)")
st.caption("Edit text/author/tags. Uncheck approve to discard. Tags are comma-separated.")

if include_debug:
    mode_counts = Counter(r.get("_mode", "unknown") or "unknown" for r in rows)
    parts = [f"{k}: {v}" for k, v in sorted(mode_counts.items(), key=lambda x: (-x[1], x[0]))]
    st.write("**Results by mode:** " + " • ".join(parts))

df = as_editor_df(rows, include_debug=bool(include_debug))

column_config = {
    "approve": st.column_config.CheckboxColumn("Approve", width="small"),
    "text": st.column_config.TextColumn("Text", width="large"),
    "author": st.column_config.TextColumn("Author", width="medium"),
    "tags_str": st.column_config.TextColumn("Tags (comma-separated)", width="medium"),
}
if include_debug:
    column_config["_mode"] = st.column_config.TextColumn("Mode", width="small")
    column_config["_kind"] = st.column_config.TextColumn("Kind", width="small")
    column_config["_author_src"] = st.column_config.TextColumn("Author src", width="small")
    column_config["_group_id"] = st.column_config.TextColumn("Group", width="small")
    column_config["_chunk_i"] = st.column_config.TextColumn("Chunk i", width="small")
    column_config["_chunk_n"] = st.column_config.TextColumn("Chunk n", width="small")

edited = st.data_editor(
    df,
    width='stretch',
    num_rows="fixed",
    column_config=column_config,
)

approve_count = int(edited["approve"].sum())
st.write(f"Approved: **{approve_count}** / {len(edited)}")

approved_preview = edited[edited["approve"] == True].copy()
if not approved_preview.empty:
    st.download_button(
        "Download approved as CSV (preview)",
        data=approved_preview.to_csv(index=False).encode("utf-8"),
        file_name="approved_quotes_preview.csv",
        mime="text/csv",
    )

save_clicked = st.button("Save approved to JSONL", disabled=(approve_count == 0))
if not save_clicked:
    st.stop()

existing_keys_now = set(load_existing_keys(jsonl_path))
global_tag_list = parse_tag_line(global_tags)

to_write: List[Dict[str, object]] = []
skipped_dupe = 0
skipped_invalid = 0

for rec in approved_preview.to_dict("records"):
    text = str(rec.get("text", "")).strip()
    if not text:
        skipped_invalid += 1
        continue

    author = str(rec.get("author", "")).strip() or default_author
    tags = parse_tag_line(str(rec.get("tags_str", "")))

    if global_tag_list:
        existing_lower = {x.lower() for x in tags}
        tags.extend([t for t in global_tag_list if t.lower() not in existing_lower])

    key = normalize_key(text)
    if key in existing_keys_now:
        skipped_dupe += 1
        continue

    to_write.append({"text": text, "author": author, "tags": tags})
    existing_keys_now.add(key)

if to_write:
    appended = append_jsonl(jsonl_path, to_write)
    st.success(f"Saved **{appended}** new quote(s) to `{jsonl_path}`.")
    load_existing_keys.clear()
else:
    st.warning("No new quotes to save after validation/dedupe.")

if skipped_dupe:
    st.info(f"Skipped duplicates already in dataset: **{skipped_dupe}**")
if skipped_invalid:
    st.info(f"Skipped invalid (empty text) rows: **{skipped_invalid}**")

st.session_state["rows"] = []
st.rerun()
