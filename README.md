# Atomos

A lightweight, regex‑based quote extraction tool with a Streamlit review UI and JSONL append‑only storage.

This project is designed to:

* Extract quotes from messy text sources (news, transcripts, quote dumps, tables, etc.)
* Infer speakers/authors when possible
* Let a human quickly review, edit, tag, and approve quotes
* Append clean `{text, author, tags[]}` records to a growing JSONL dataset

---

## Project Layout

```
atomos/
  app.py            # Streamlit UI + review + JSONL persistence
  parser_core.py    # Pure quote extraction + attribution engine
  requirements.txt
  Dockerfile
  README.md
```

* `parser_core.py` contains all parsing logic and can be tested independently.
* `app.py` is only UI + caching + file I/O (kept thin on purpose).

---

## Running with Docker

### Build the image

```bash
docker build -t atomos .
```

### Run the container

```bash
docker run -p 8501:8501 -v $(pwd)/data:/data atomos
```

Then open:

```
http://localhost:8501
```

All quotes will be saved to:

```
./data/quotes.jsonl
```

---

## Environment variables

You can override default storage paths:

* `DATA_DIR` (default: `/data`)
* `JSONL_PATH` (default: `/data/quotes.jsonl`)

Example:

```bash
export JSONL_PATH=/tmp/my_quotes.jsonl
streamlit run app.py
```

---

## Features

* Extracts quoted spans (supports smart quotes + nesting)
* Transcript speaker labels (`HOST: text`)
* Quote lists with author carry‑forward (Goodreads / numbered lists)
* Table / TSV row extraction
* Heuristic speaker attribution (`Name said …` before/after quotes)
* Paragraph‑level carry‑forward of speakers
* Human‑in‑the‑loop review UI
* Append‑only JSONL output with deduplication

---

## Output format (JSONL)

Each approved quote is written as one JSON object per line:

```json
{"text": "The only limit to our realization of tomorrow is our doubts of today.", "author": "Franklin D. Roosevelt", "tags": ["inspiration", "future"]}
```

---

## Notes

* `max_len` is intentionally fixed at 240 characters by design.
* The parser is regex‑based and heuristic: accuracy improves with human review.
* `parser_core.py` and `app.py` are intentionally kept separate for easier debugging and benchmarking.

---

