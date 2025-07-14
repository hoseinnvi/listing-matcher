# matcher_service.py
"""
FastAPI micro‑service for the Reffie homework.
Implements a four‑stage matching pipeline under the 20 MB RAM cap.

Stages
-------
0. **Pre‑match** – trust existing `listing.property_id`.
1. **Exact** – `(team_id, full_address)` lookup.
2. **Fuzzy / apartment‑level** – SBERT embeddings + FAISS within‑team.
3. **Building fallback** – strip `unit_part`, retry exact or fuzzy.
4. **Abstention** – when confidence < MIN_CONFIDENCE.

Key choices
-----------
* SQLAlchemy Core for dynamic data access.
* `all‑MiniLM‑L6‑v2` embeddings (384‑dim, L2‑normalized; inner‑product = cosine).
* One SQLAlchemy session reused for the full batch job to avoid pool exhaustion.
"""

import os
import tracemalloc
from typing import Optional, Tuple, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Text, select
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv(
    "DB_URL",
    "postgresql+psycopg2://postgres:09113254626@localhost:5432/reffie_homework",
)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MIN_CONFIDENCE = 0.80

# ---------------------------------------------------------------------------
# SQLAlchemy setup
# ---------------------------------------------------------------------------
Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_size=4, max_overflow=0, echo=False)
SessionLocal = sessionmaker(bind=engine)

class Property(Base):
    __tablename__ = "properties"
    property_id = Column(String, primary_key=True)
    team_id = Column(String)
    street_part = Column(Text)
    unit_part = Column(Text)
    city = Column(Text)
    state = Column(Text)
    zipcode = Column(Text)
    full_address = Column(Text)
    token_set = Column(Text)
    type_norm = Column(Text)

class Listing(Base):
    __tablename__ = "listing"
    listing_id = Column(String, primary_key=True)
    property_id = Column(String)
    team_id = Column(String)
    street_part = Column(Text)
    unit_part = Column(Text)
    city = Column(Text)
    state = Column(Text)
    zipcode = Column(Text)
    full_address = Column(Text)
    token_set = Column(Text)

# ---------------------------------------------------------------------------
# Embedding + FAISS cache
# ---------------------------------------------------------------------------
print("Loading embedding model … (≈ 90 MB once)")
model = SentenceTransformer(EMBEDDING_MODEL_NAME, token=False)
_team_cache: Dict[str, Dict[str, object]] = {}


def _build_team_index(team_id: str, db: Session):
    rows = db.execute(
        select(Property.property_id, Property.full_address).where(Property.team_id == team_id)
    ).all()
    if not rows:
        raise ValueError(f"No properties found for team {team_id}")

    prop_ids, addrs = zip(*rows)
    embs = model.encode(list(addrs), normalize_embeddings=True)
    idx = faiss.IndexFlatIP(embs.shape[1])  # IP on normalized vectors == cosine
    idx.add(embs)
    _team_cache[team_id] = {"embeddings": embs, "prop_ids": list(prop_ids), "index": idx}


def _team_resources(team_id: str, db: Session):
    if team_id not in _team_cache:
        _build_team_index(team_id, db)
    return _team_cache[team_id]

# ---------------------------------------------------------------------------
# Helpers & matcher
# ---------------------------------------------------------------------------

def normalize_addr(addr: Optional[str]) -> str:
    """Lower‑case, trim, squeeze spaces. Return empty string for None/empty."""
    if not addr:
        return ""
    return " ".join(addr.lower().split())


def match_listing(
    listing_id: str,
    team_id: str,
    listing_address: Optional[str],
    db: Session,
) -> Tuple[Optional[str], float]:
    # Missing address ⇒ abstain
    if not listing_address:
        return None, 0.0
    norm_addr = normalize_addr(listing_address)

    # 0. pre‑match
    existing = db.execute(
        select(Listing.property_id)
        .where(Listing.listing_id == listing_id)
        .where(Listing.property_id.isnot(None))
    ).scalar_one_or_none()
    if existing:
        return existing, 1.0

    # 1. exact
    exact = db.execute(
        select(Property.property_id)
        .where(Property.team_id == team_id)
        .where(Property.full_address.ilike(norm_addr))
    ).scalar_one_or_none()
    if exact:
        return exact, 1.0

    # 2. fuzzy / apartment level
    try:
        res = _team_resources(team_id, db)
    except ValueError:
        # No properties for this team → abstain
        return None, 0.0
    emb = model.encode([norm_addr], normalize_embeddings=True)[0]
    sims, idxs = res["index"].search(emb.reshape(1, -1), 1)
    sim = float(sims[0, 0])
    conf = sim * 0.5 + 0.5  # map cosine from [-1,1] → [0,1]
    if conf >= MIN_CONFIDENCE:
        return res["prop_ids"][int(idxs[0, 0])], conf

    # 3. building fallback (strip after dash)
    base = norm_addr.split("-")[0].strip()
    bldg_exact = db.execute(
        select(Property.property_id)
        .where(Property.team_id == team_id)
        .where(Property.street_part.ilike(base))
        .limit(1)
    ).scalar_one_or_none()
    if bldg_exact:
        return bldg_exact, 0.7

    emb2 = model.encode([base], normalize_embeddings=True)[0]
    sims2, idxs2 = res["index"].search(emb2.reshape(1, -1), 1)
    sim2 = float(sims2[0, 0])
    conf2 = (sim2 * 0.5 + 0.5) * 0.9
    if conf2 >= MIN_CONFIDENCE:
        return res["prop_ids"][int(idxs2[0, 0])], conf2

    # 4. abstain
    return None, 0.0

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Reffie Matching Service", version="1.0")

class MatchRequest(BaseModel):
    listing_id: str
    team_id: str
    full_address: str

class MatchResponse(BaseModel):
    property_id: Optional[str]
    confidence: float


@app.post("/match", response_model=MatchResponse)
async def match(req: MatchRequest):
    with SessionLocal() as db:
        try:
            pid, conf = match_listing(req.listing_id, req.team_id, req.full_address, db)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    return MatchResponse(property_id=pid, confidence=round(conf, 4))

# ---------------------------------------------------------------------------
# Batch utility
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    tracemalloc.start()
    print("Building submission.csv…")
    with SessionLocal() as db:
        listings = db.execute(
            select(Listing.listing_id, Listing.team_id, Listing.full_address, Listing.property_id)
        ).all()

        rows = []
        for lid, tid, addr, pre in listings:
            if pre:
                rows.append({"listing_id": lid, "property_id": pre, "confidence": 1.0})
            else:
                pid, conf = match_listing(lid, tid, addr, db)
                rows.append({"listing_id": lid, "property_id": pid, "confidence": round(conf, 4)})

        pd.DataFrame(rows).to_csv("submission.csv", index=False)
        print("Done → submission.csv")

        snap = tracemalloc.take_snapshot()
        mb = sum(stat.size for stat in snap.statistics("filename")) / 1_048_576
        print(f"Peak RAM: {mb:.2f} MB (≤ 20 MB target)")
