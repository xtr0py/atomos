# Atomos

A lightweight, regex‑based quote extraction tool with a Streamlit review UI and JSONL append‑only storage.

---

## Key Features

- **Finds quotes automatically** from articles, transcripts, and pasted text (handles smart quotes and nested quotes)  
- **Figures out who said each quote** using nearby context like “Name said” and by carrying the speaker forward within a paragraph  
- **Extracts quotes from transcripts and lists** (lines like `HOST: text`, numbered lists, and pages with repeated quotes and authors)  
- **Lets you review and edit everything by hand** in a clean web interface before saving  
- **Prevents duplicates automatically** by detecting quotes that already exist in your dataset  
- **Saves clean, structured data** as append-only JSONL (`text`, `author`, `tags`) ready for analysis or training 

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

## Output format (JSONL)

Each approved quote is written as one JSON object per line:

```json
{"text": "The only limit to our realization of tomorrow is our doubts of today.", "author": "Franklin D. Roosevelt", "tags": ["inspiration", "future"]}
```
