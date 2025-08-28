

Match free‑form **listings** to canonical **properties** with high precision, < 20 MB inference RAM, and a FastAPI micro‑service.

---

## Table of Contents

1. [Quick start](#quick-start)
2. [Setup](#setup)
3. [Database initialization](#database-initialization)
4. [Running the API](#running-the-api)
5. [Batch `submission.csv`](#batch-submissioncsv)
6. [Test suite](#test-suite)
7. [Design highlights](#design-highlights)
8. [Memory & performance](#memory--performance)
9. [Alternatives considered](#alternatives-considered)
10. [License](#license)

---

## Quick start

```bash
# 0️⃣  clone + venv
git clone https://github.com/hoseinnvi/reffie-matcher.git
cd reffie-matcher
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# ①  create & load a local Postgres DB (defaults: user=postgres, db=reffie_homework)
psql -U postgres -f Database_creating/db_init.sql

# ②  launch the API
uvicorn app.matcher_service:app --host 0.0.0.0 --port 8000

# ③  hit the endpoint
curl -X POST http://localhost:8000/match \
  -H "Content-Type: application/json" \
  -d '{
        "listing_id":"demo-id",
        "team_id":"demo-team",
        "full_address":"1341 Spring Creek Dr Provo UT 84606"
      }'
```

---

## Setup

| Dependency | Version / Notes                                                         |
| ---------- | ----------------------------------------------------------------------- |
| Python     | 3.10 +                                                                  |
| PostgreSQL | 14 +                                                                    |
| FAISS      | CPU wheel in `requirements.txt` (swap for `faiss-gpu` if you have CUDA) |

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Database initialization

`Database_creating/db_init.sql` does the heavy lifting:

* Drops & recreates `properties` and `listing` tables
* Adds all required columns & indexes
* Bulk‑loads the cleaned CSVs via `\copy`

```bash
psql -U postgres -d reffie_homework -f app/db_init.sql
```

> **Environment variable**: override connection string with
> `export DB_URL="postgresql+psycopg2://user:pass@host/db"`

---

## Running the API

```bash
uvicorn app.matcher_service:app --reload
```

* **POST `/match`**

  ```jsonc
  {
    "listing_id": "uuid-string",
    "team_id"   : "uuid-string",
    "full_address": "1341 Spring Creek Dr Provo UT 84606"
  }
  ```
* **Response**

  ```jsonc
  {
    "property_id": "uuid-string | null",
    "confidence" : 0.0-1.0
  }
  ```

Interactive docs: open **[http://localhost:8000/docs](http://localhost:8000/docs)**.

---

## Batch `submission.csv`

Re‑create the autograder file in one command:

```bash
python -m app.matcher_service          # runs the __main__ block
# → submission.csv with columns: listing_id, property_id, confidence
```

---

## Test suite

```bash
pytest -q
```

Tests cover:

* Exact match returns `confidence == 1.0`
* Fuzzy match (minor typo) returns `confidence ≥ 0.8`
* Orphan team ⇒ `(null, 0.0)` abstention
* Peak RAM ≤ 20 MB (`tracemalloc`)

---

## Design highlights

| Stage           | Logic                                               | Confidence  |
| --------------- | --------------------------------------------------- | ----------- |
| **0 Pre‑match** | Trust `listing.property_id` already in DB           | **1.0**     |
| **1 Exact**     | `team_id` + exact `full_address` via index          | **1.0**     |
| **2 Fuzzy**     | SBERT `MiniLM‑L6‑v2` → FAISS `IndexFlatIP` (cosine) | `(cos+1)/2` |
| **3 Building**  | Strip `unit_part`; retry exact→fuzzy                | 0.7‑0.9     |
| **4 Abstain**   | If `confidence < 0.8`                               | 0.0         |

* Single SQLAlchemy session per request/batch → no pool exhaustion.
* Per‑team FAISS cache (\~6 MB each) built lazily in RAM.
* Peak RAM (batch) **18.9 MB**.

Full rationale in **Approach.md**.

---

## Memory & performance


| Metric                     | Value       |
| -------------------------- | ----------- |
| Peak Python RAM            | **18.9 MB** |
| Median exact‑match latency | 2 ms        |
| Median fuzzy‑match latency | 6 ms        |

---

## Alternatives considered

| Decision             | Alternative                  | Why chosen                                          |
| -------------------- | ---------------------------- | --------------------------------------------------- |
| SBERT + FAISS        | `pg_trgm` trigram similarity | +5‑8 % recall on typos within RAM cap               |
| Per‑team index       | One global index             | Simpler filtering; predictable ≤20 MB               |
| Inner‑product cosine | L2 or ANN‑HNSW               | IP on normalized vectors is fastest & deterministic |

More discussion in *Approach.md*.

---

## License

MIT © 2025 **Hosein Navaei Moakhkhar**
*Provided for the Reffie ; not for production use without permission.*
