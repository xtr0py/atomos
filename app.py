import os
import json
import hashlib
from typing import List, Dict, Set

import pandas as pd
import streamlit as st

from parser_core import extract_quotes, normalize_key, parse_tag_line

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

@st.cache_data(show_spinner=False)
def cached_parse(
    source_text: str,
    default_author: str,
    min_len: int,
    max_newlines: int,
    max_sentences: int,
    enable_speaker_labels: bool,
    enable_quote_lists: bool,
    enable_table_rows: bool,
    enable_paragraph_carry: bool,
) -> List[Dict[str, object]]:
    parsed = extract_quotes(
        source_text,
        default_author=default_author,
        min_len=min_len,
        max_len=240,  # fixed
        max_newlines=max_newlines,
        max_sentences=max_sentences,
        enable_speaker_labels=enable_speaker_labels,
        enable_quote_lists=enable_quote_lists,
        enable_table_rows=enable_table_rows,
        enable_paragraph_carry=enable_paragraph_carry,
    )
    for r in parsed:
        r["approve"] = True
        r["tags_str"] = tags_to_str(r.get("tags", []))
    return parsed

def as_editor_df(rows: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["approve", "text", "author", "tags_str"])
    for col in ["approve", "text", "author", "tags_str"]:
        if col not in df.columns:
            df[col] = True if col == "approve" else ""
    return df[["approve", "text", "author", "tags_str"]]

# -----------------------------
# Sidebar settings
# -----------------------------

with st.sidebar:
    st.header("Settings")

    jsonl_path = st.text_input("JSONL output path", value=DEFAULT_JSONL_PATH)
    default_author = st.text_input("Default author (when none found)", value="Unknown")

    st.subheader("Minimal quote filters")
    min_len = st.number_input("Min length", min_value=1, max_value=500, value=30)
    st.number_input("Max length", min_value=30, max_value=240, value=240, disabled=True)
    st.caption("Max length is fixed at 240 by design (per project requirement).")
    max_sentences = st.number_input("Max sentences", min_value=1, max_value=10, value=2)
    max_newlines = st.number_input("Max newlines", min_value=0, max_value=10, value=1)

    st.subheader("Extraction modes")
    enable_speaker_labels = st.toggle("Transcript speaker labels (LABEL: text)", value=True)
    enable_quote_lists = st.toggle("Quote-list mode (Number X / author carry / Goodreads)", value=True)
    enable_table_rows = st.toggle("Table/TSV row mode (quote ⟂ author ⟂ ...)", value=True)
    enable_paragraph_carry = st.toggle("Carry-forward author within paragraph", value=True)

    st.subheader("Tag helpers")
    global_tags = st.text_input("Global tags (comma-separated, applied on save)", value="")

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
        max_newlines=int(max_newlines),
        max_sentences=int(max_sentences),
        enable_speaker_labels=bool(enable_speaker_labels),
        enable_quote_lists=bool(enable_quote_lists),
        enable_table_rows=bool(enable_table_rows),
        enable_paragraph_carry=bool(enable_paragraph_carry),
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
st.caption("Edit text/author/tags. Uncheck approve to discard. Tags are comma-separated; extracted tags may already be present.")

df = as_editor_df(rows)

edited = st.data_editor(
    df,
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "approve": st.column_config.CheckboxColumn("Approve", width="small"),
        "text": st.column_config.TextColumn("Text", width="large"),
        "author": st.column_config.TextColumn("Author", width="medium"),
        "tags_str": st.column_config.TextColumn("Tags (comma-separated)", width="medium"),
    },
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

    # Apply global tags (merge, case-insensitive)
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
    load_existing_keys.clear()  # refresh dataset count next run
else:
    st.warning("No new quotes to save after validation/dedupe.")

if skipped_dupe:
    st.info(f"Skipped duplicates already in dataset: **{skipped_dupe}**")
if skipped_invalid:
    st.info(f"Skipped invalid (empty text) rows: **{skipped_invalid}**")

st.session_state["rows"] = []
st.rerun()
