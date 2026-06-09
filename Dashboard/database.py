"""
utils/database.py
SQLite-backed feature store for the SEM dashboard.
"""
from __future__ import annotations

import sqlite3
import json
import pickle
import base64
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DB_PATH, EXPORT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _np_to_b64(arr: np.ndarray) -> str:
    return base64.b64encode(pickle.dumps(arr)).decode("ascii")

def _b64_to_np(s: str) -> np.ndarray:
    return pickle.loads(base64.b64decode(s.encode("ascii")))


# ─────────────────────────────────────────────────────────────────────────────
# DB
# ─────────────────────────────────────────────────────────────────────────────

class SEMDatabase:
    """Thin wrapper around SQLite for SEM image feature storage."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(EXPORT_DIR, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename        TEXT NOT NULL,
                    category        TEXT DEFAULT 'Unknown',
                    upload_time     TEXT NOT NULL,
                    morph_features  TEXT,
                    graph_features  TEXT,
                    deep_embedding  TEXT,
                    simclr_embedding TEXT,
                    hybrid_vector   TEXT,
                    thumbnail       TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS batch_runs (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_time    TEXT NOT NULL,
                    n_images    INTEGER,
                    notes       TEXT
                )
            """)
            conn.commit()

    # ── Write ──

    def insert_image(
        self,
        filename: str,
        category: str,
        morph: Dict[str, float],
        graph: Dict[str, float],
        deep_emb: np.ndarray,
        simclr_emb: np.ndarray,
        hybrid_vec: Optional[np.ndarray] = None,
        thumbnail: Optional[np.ndarray] = None,
    ) -> int:
        thumb_b64 = _np_to_b64(thumbnail) if thumbnail is not None else None
        hyb_b64   = _np_to_b64(hybrid_vec) if hybrid_vec is not None else None
        with self._conn() as conn:
            cur = conn.execute("""
                INSERT INTO images
                  (filename, category, upload_time,
                   morph_features, graph_features,
                   deep_embedding, simclr_embedding,
                   hybrid_vector, thumbnail)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                filename, category, datetime.utcnow().isoformat(),
                json.dumps(morph), json.dumps(graph),
                _np_to_b64(deep_emb), _np_to_b64(simclr_emb),
                hyb_b64, thumb_b64,
            ))
            conn.commit()
            return cur.lastrowid

    # ── Read ──

    def get_all_metadata(self) -> pd.DataFrame:
        with self._conn() as conn:
            df = pd.read_sql("SELECT id, filename, category, upload_time FROM images", conn)
        return df

    def get_morph_dataframe(self) -> pd.DataFrame:
        with self._conn() as conn:
            rows = conn.execute("SELECT id, filename, category, morph_features FROM images").fetchall()
        records = []
        for row_id, fn, cat, mf in rows:
            rec = json.loads(mf) if mf else {}
            rec["id"] = row_id
            rec["filename"] = fn
            rec["category"] = cat
            records.append(rec)
        return pd.DataFrame(records)

    def get_all_embeddings(self, kind: str = "hybrid") -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Returns (embedding_matrix, categories, filenames).
        kind: 'deep' | 'simclr' | 'hybrid'
        """
        col_map = {
            "deep"  : "deep_embedding",
            "simclr": "simclr_embedding",
            "hybrid": "hybrid_vector",
        }
        col = col_map.get(kind, "hybrid_vector")
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT category, filename, {col} FROM images WHERE {col} IS NOT NULL"
            ).fetchall()
        if not rows:
            return np.empty((0,)), [], []
        cats, fns, vecs = [], [], []
        for cat, fn, blob in rows:
            cats.append(cat)
            fns.append(fn)
            vecs.append(_b64_to_np(blob))
        return np.vstack(vecs), cats, fns

    def get_thumbnails(self) -> List[Tuple[int, str, Optional[np.ndarray]]]:
        with self._conn() as conn:
            rows = conn.execute("SELECT id, filename, thumbnail FROM images").fetchall()
        result = []
        for row_id, fn, thumb in rows:
            arr = _b64_to_np(thumb) if thumb else None
            result.append((row_id, fn, arr))
        return result

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    def delete_image(self, row_id: int):
        with self._conn() as conn:
            conn.execute("DELETE FROM images WHERE id=?", (row_id,))
            conn.commit()

    def clear(self):
        with self._conn() as conn:
            conn.execute("DELETE FROM images")
            conn.commit()

    # ── Export ──

    def export_morphology_csv(self) -> str:
        df = self.get_morph_dataframe()
        path = os.path.join(EXPORT_DIR, f"sem_morphology_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(path, index=False)
        return path

    def export_embeddings_npz(self, kind: str = "hybrid") -> str:
        vecs, cats, fns = self.get_all_embeddings(kind)
        path = os.path.join(EXPORT_DIR, f"sem_{kind}_embeddings_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.npz")
        np.savez(path, embeddings=vecs,
                 categories=np.array(cats, dtype=object),
                 filenames=np.array(fns, dtype=object))
        return path

    def export_full_json(self) -> str:
        df_meta  = self.get_all_metadata()
        df_morph = self.get_morph_dataframe()
        merged   = df_meta.merge(df_morph.drop(columns=["filename", "category"], errors="ignore"),
                                 on="id", how="left")
        path = os.path.join(EXPORT_DIR, f"sem_full_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
        merged.to_json(path, orient="records", indent=2)
        return path
